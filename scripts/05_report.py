"""Turn the evaluation JSON into the tables and figures in results/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pii_pipeline.entities import TYPES_WITHOUT_CORPUS_SUPPORT

# Slots 1-3 of the validated categorical theme. Validated for all pairs in light
# mode; aqua sits below 3:1 contrast, so every chart carries direct value labels
# (the "relief" requirement) and a CSV/markdown table alongside.
PALETTE = {
    "baseline_rules_presidio": "#2a78d6",
    "finetuned_model": "#eb6834",
    "ensemble": "#1baf7a",
}
LABELS = {
    "rules": "Rules (regex + checksums)",
    "presidio": "Presidio",
    "baseline_rules_presidio": "Baseline (rules + Presidio)",
    "finetuned_model": "Fine-tuned model",
    "ensemble": "Ensemble (baseline + model)",
}
MAIN_SYSTEMS = ["baseline_rules_presidio", "finetuned_model", "ensemble"]
INK, INK_2, GRID = "#1a1a1a", "#555555", "#e2e2e2"


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def build_tables(results: dict[str, Any], out_dir: Path) -> pd.DataFrame:
    rows = []
    for ts_name, ts in results["test_sets"].items():
        for sys_name, entry in ts["systems"].items():
            ex, pa = entry["exact"], entry["partial"]
            rows.append(
                {
                    "test_set": ts_name,
                    "system": LABELS.get(sys_name, sys_name),
                    "system_key": sys_name,
                    "P_exact": ex["micro"]["precision"],
                    "R_exact": ex["micro"]["recall"],
                    "F1_exact": ex["micro"]["f1"],
                    "F1_partial": pa["micro"]["f1"],
                    "F1_macro": ex["macro"]["f1"],
                    "entity_leakage": pa["privacy"]["entity_leakage_rate"],
                    "over_redaction": pa["privacy"]["over_redaction_rate"],
                    "docs_per_sec": entry["docs_per_second"],
                }
            )
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "comparison.csv", index=False)

    lines = [
        "# Baseline vs fine-tuned model comparison",
        "",
        f"> {results['note']}",
        "",
    ]
    for ts_name in results["test_sets"]:
        sub = df[df.test_set == ts_name].drop(columns=["test_set", "system_key"])
        lines += [
            f"## Test set: `{ts_name}` "
            f"({results['test_sets'][ts_name]['n_documents']} documents)",
            "",
            sub.to_markdown(index=False, floatfmt=".4f"),
            "",
        ]

    # Per-type detail for the strongest system on the out-of-domain set.
    ood = results["test_sets"].get("synthetic_documents", {}).get("systems", {})
    if ood:
        best = max(ood, key=lambda s: ood[s]["partial"]["micro"]["f1"])
        by_type = ood[best]["partial"]["by_type"]
        rows_t = [
            {
                "type": t,
                "P": v["precision"],
                "R": v["recall"],
                "F1": v["f1"],
                "support": v["support"],
            }
            for t, v in sorted(by_type.items())
        ]
        lines += [
            f"## Per-type detail — {LABELS.get(best, best)} on synthetic documents "
            "(partial match, IoU 0.5)",
            "",
            pd.DataFrame(rows_t).to_markdown(index=False, floatfmt=".4f"),
            "",
        ]
    (out_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return df


def fig_f1_by_type(results: dict[str, Any], test_set: str, out_path: Path) -> None:
    systems = results["test_sets"][test_set]["systems"]
    available = [s for s in MAIN_SYSTEMS if s in systems]
    if not available:
        return
    types = sorted(
        {t for s in available for t in systems[s]["partial"]["by_type"]}
    )
    y = np.arange(len(types))
    height = 0.78 / len(available)

    fig, ax = plt.subplots(figsize=(9.5, 0.62 * len(types) + 2.6))
    for i, sys_name in enumerate(available):
        by_type = systems[sys_name]["partial"]["by_type"]
        values = [by_type.get(t, {}).get("f1", 0.0) for t in types]
        offset = (i - (len(available) - 1) / 2) * height
        bars = ax.barh(
            y + offset, values, height=height * 0.88,
            color=PALETTE[sys_name], label=LABELS[sys_name], zorder=3,
        )
        # Direct labels satisfy the contrast "relief" requirement.
        for bar, val in zip(bars, values):
            ax.text(
                val + 0.012, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left", fontsize=7.4, color=INK_2,
            )

    labels = [
        f"{t} *" if t in TYPES_WITHOUT_CORPUS_SUPPORT else t for t in types
    ]
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("F1 (partial match, IoU 0.5)", color=INK_2, fontsize=9)
    ax.set_title(
        f"F1 per entity type — {test_set}",
        color=INK, fontsize=12, fontweight="bold", loc="left", pad=34,
    )
    ax.invert_yaxis()
    _style(ax)
    # Above the plot area: at the bottom right it collided with full-length bars.
    ax.legend(
        frameon=False, fontsize=8.5, labelcolor=INK_2, ncol=len(available),
        loc="lower left", bbox_to_anchor=(0, 1.005),
    )
    fig.text(
        0.01, 0.005,
        "* class absent from the training corpus: reachable only by the rule engine",
        fontsize=7.2, color=INK_2,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, facecolor="white")
    plt.close(fig)


def fig_privacy(results: dict[str, Any], out_path: Path) -> None:
    test_sets = list(results["test_sets"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    metrics = [
        ("entity_leakage_rate", "PII leakage (lower is better)"),
        ("over_redaction_rate", "Over-redaction (lower is better)"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        x = np.arange(len(test_sets))
        available = [
            s for s in MAIN_SYSTEMS
            if all(s in results["test_sets"][ts]["systems"] for ts in test_sets)
        ]
        width = 0.8 / max(1, len(available))
        for i, sys_name in enumerate(available):
            values = [
                results["test_sets"][ts]["systems"][sys_name]["partial"]["privacy"][metric]
                for ts in test_sets
            ]
            offset = (i - (len(available) - 1) / 2) * width
            bars = ax.bar(
                x + offset, values, width=width * 0.88,
                color=PALETTE[sys_name], label=LABELS[sys_name], zorder=3,
            )
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, val,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.4, color=INK_2,
                )
        ax.set_xticks(x, test_sets, fontsize=8.5)
        ax.set_title(title, color=INK, fontsize=10.5, fontweight="bold", loc="left", pad=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8.2, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, facecolor="white")
    plt.close(fig)


def fig_reliability(calibration: dict[str, Any], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.6, linestyle="--", zorder=1)
    ax.text(0.66, 0.60, "perfect calibration", fontsize=7.6, color="#999999", rotation=34)

    total = max(1, sum(b["count"] for b in calibration["reliability_before"]))
    for key, color, label, ece_key in [
        ("reliability_before", "#2a78d6", "Raw score", "ece_before"),
        ("reliability_after", "#1baf7a", "Grouped calibration", "ece_after"),
    ]:
        bins = [b for b in calibration[key] if b["count"] > 0]
        if not bins:
            continue
        xs = [b["mean_confidence"] for b in bins]
        ys = [b["accuracy"] for b in bins]
        # Marker area tracks how many predictions fall in the bin. Without this
        # a bin holding 20 predictions looks as important as one holding 1,500,
        # and the chart contradicts the (count-weighted) ECE printed beside it.
        sizes = [40 + 900 * (b["count"] / total) for b in bins]
        ax.plot(xs, ys, linewidth=1.3, color=color, alpha=0.55, zorder=2)
        ax.scatter(
            xs, ys, s=sizes, color=color, zorder=3, alpha=0.9,
            edgecolors="white", linewidths=1.5,
            label=f"{label} (ECE {calibration[ece_key]:.3f})",
        )
    ax.set_xlabel("Mean confidence in bin", color=INK_2, fontsize=9)
    ax.set_ylabel("Observed precision", color=INK_2, fontsize=9)
    ax.set_title(
        "Reliability diagram", color=INK, fontsize=12, fontweight="bold",
        loc="left", pad=14,
    )
    if "ece_after_global_platt" in calibration:
        ax.text(
            0.02, 0.06,
            f"Global Platt (ungrouped): ECE {calibration['ece_after_global_platt']:.3f}\n"
            "no gain: the raw curve is non-monotonic",
            fontsize=7.8, color=INK_2, transform=ax.transAxes,
        )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK_2,
              scatterpoints=1, markerscale=0.45)
    ax.text(
        0.98, 0.02, "marker area is proportional to\nthe number of predictions in the bin",
        fontsize=7.2, color=INK_2, transform=ax.transAxes, ha="right",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, facecolor="white")
    plt.close(fig)


def fig_review_queue(queue_stats: dict[str, Any], out_path: Path) -> None:
    order = ["auto_redact", "review", "discard"]
    names = {
        "auto_redact": "Automatic redaction",
        "review": "Review queue",
        "discard": "Discarded",
    }
    colors = {"auto_redact": "#1baf7a", "review": "#eb6834", "discard": "#2a78d6"}
    values = [queue_stats["decisions"].get(k, 0) for k in order]
    total = sum(values) or 1

    fig, ax = plt.subplots(figsize=(8.4, 3.1))
    left = 0.0
    for key, val in zip(order, values):
        ax.barh([0], [val], left=left, color=colors[key], height=0.5, zorder=3)
        if val / total > 0.03:
            ax.text(
                left + val / 2, 0,
                f"{names[key]}\n{val} ({val / total:.0%})",
                ha="center", va="center", fontsize=8.6, color="white", fontweight="bold",
            )
        left += val
    ax.set_xlim(0, total)
    ax.set_yticks([])
    ax.set_xlabel("Detected entities", color=INK_2, fontsize=9)
    ax.set_title(
        f"Routing of {total} entities across {queue_stats['n_documents']} documents",
        color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=12,
    )
    ax.spines[:].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=Path("results/metrics/evaluation.json"))
    parser.add_argument("--calibration", type=Path, default=Path("results/metrics/calibration.json"))
    parser.add_argument("--queue-stats", type=Path, default=Path("results/metrics/queue_stats.json"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.evaluation.read_text(encoding="utf-8"))
    df = build_tables(results, args.metrics_dir)

    for ts_name in results["test_sets"]:
        fig_f1_by_type(results, ts_name, args.figures_dir / f"f1_by_type_{ts_name}.png")
    fig_privacy(results, args.figures_dir / "privacy_tradeoff.png")

    if args.calibration.exists():
        fig_reliability(
            json.loads(args.calibration.read_text(encoding="utf-8")),
            args.figures_dir / "reliability_diagram.png",
        )
    if args.queue_stats.exists():
        fig_review_queue(
            json.loads(args.queue_stats.read_text(encoding="utf-8")),
            args.figures_dir / "review_queue.png",
        )

    print(df.to_string(index=False))
    print(f"\ntables -> {args.metrics_dir}  figures -> {args.figures_dir}")


if __name__ == "__main__":
    main()
