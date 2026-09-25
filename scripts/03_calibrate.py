"""Fit the confidence calibrator on held-out validation documents.

Detector scores are not probabilities: a rule that says 0.9 and a model that
says 0.9 are not right equally often. The review-queue thresholds only mean
something once those scores have been mapped onto observed correctness.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

from pii_pipeline.data.build_dataset import read_jsonl
from pii_pipeline.evaluation.metrics import match_spans, merge_adjacent_spans
from pii_pipeline.evaluation.systems import build_systems, to_spans
from pii_pipeline.scoring import (
    ConfidenceCalibrator,
    expected_calibration_error,
    reliability_bins,
)

logger = logging.getLogger(__name__)


def _apply_global(calibrator: ConfidenceCalibrator, score: float) -> float:
    """Force the global Platt curve onto a score, ignoring group fits."""
    z = calibrator.slope * math.log(
        min(max(score, 1e-6), 1 - 1e-6) / (1 - min(max(score, 1e-6), 1 - 1e-6))
    ) + calibrator.intercept
    return 1.0 / (1.0 + math.exp(-z))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/distilbert-pii-ner/final"))
    parser.add_argument("--out", type=Path, default=Path("models/calibrator.json"))
    parser.add_argument("--report", type=Path, default=Path("results/metrics/calibration.json"))
    parser.add_argument("-n", "--num-docs", type=int, default=600)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    records = read_jsonl(args.data_dir / "validation.jsonl")[: args.num_docs]
    ensemble = build_systems(args.model_dir)["ensemble"]

    scores: list[float] = []
    correct: list[int] = []
    group_keys: list[str] = []
    for i, rec in enumerate(records):
        if i and i % 100 == 0:
            logger.info("calibration: %d/%d documents", i, len(records))
        # Same granularity normalisation the evaluation uses, so the calibrator
        # is not fitted on an artefact of the corpus's annotation style.
        pred = merge_adjacent_spans(to_spans(ensemble(rec["text"], rec["lang"])), rec["text"])
        gold = merge_adjacent_spans(rec["entities"], rec["text"])
        _, _, extra = match_spans(gold, pred, mode="partial", iou_threshold=args.iou)
        matched_pred = {pi for pi in range(len(pred))} - set(extra)
        for pi, span in enumerate(pred):
            scores.append(span["score"])
            correct.append(1 if pi in matched_pred else 0)
            group_keys.append(span.get("group", "global"))

    before = expected_calibration_error(scores, correct)

    # Ablation: what a single global Platt curve would do. It is applied
    # unconditionally here -- the production calibrator deliberately leaves
    # ungrouped scores alone, so its `transform` cannot be reused to measure this.
    global_only = ConfidenceCalibrator().fit(scores, correct)
    ece_global = expected_calibration_error(
        [_apply_global(global_only, s) for s in scores], correct
    )

    calibrator = ConfidenceCalibrator().fit(scores, correct, group_keys)
    calibrated = [calibrator.transform(s, g) for s, g in zip(scores, group_keys)]
    after = expected_calibration_error(calibrated, correct)

    calibrator.save(args.out)
    payload = {
        "n_documents": len(records),
        "n_predictions": len(scores),
        "empirical_precision": round(sum(correct) / len(correct), 4) if correct else 0.0,
        "ece_before": round(before, 4),
        "ece_after_global_platt": round(ece_global, 4),
        "ece_after": round(after, 4),
        "why_grouped": (
            "A single global Platt fit barely moves the ECE because the reliability "
            "curve is non-monotonic: Presidio's spaCy NER emits a constant 0.85 for "
            "every PERSON/LOCATION and is mostly wrong on this corpus, while a 0.85 "
            "from the rule engine is usually right. No monotonic mapping can correct "
            "both at once, so the calibrator is fitted per (detector set, entity type)."
        ),
        "slope": calibrator.slope,
        "intercept": calibrator.intercept,
        "fitted": calibrator.fitted,
        "n_groups_calibrated": len(calibrator.groups),
        "groups": {k: [round(v[0], 4), round(v[1], 4)] for k, v in sorted(calibrator.groups.items())},
        "reliability_before": reliability_bins(scores, correct),
        "reliability_after": reliability_bins(calibrated, correct),
        "reliability_global_platt": reliability_bins(
            [_apply_global(global_only, s) for s in scores], correct
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"predicciones: {len(scores)} | ECE crudo {before:.4f} -> Platt global {ece_global:.4f} "
        f"-> Platt por grupo {after:.4f} | grupos calibrados: {len(calibrator.groups)} "
        f"| precision empirica: {payload['empirical_precision']:.4f}"
    )


if __name__ == "__main__":
    main()
