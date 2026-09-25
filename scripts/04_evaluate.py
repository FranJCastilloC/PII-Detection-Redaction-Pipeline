"""Compare every detection system on both test sets.

Two test sets on purpose:

``ai4privacy``   held-out slice of the training distribution (short sentences).
``synthetic``    full business documents built by a different generator, never
                 seen during training -- this is the domain-shift check, and the
                 number that actually predicts production behaviour.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from pii_pipeline.data.build_dataset import read_jsonl
from pii_pipeline.entities import TYPES_WITHOUT_CORPUS_SUPPORT
from pii_pipeline.evaluation.metrics import evaluate
from pii_pipeline.evaluation.systems import build_systems, to_spans
from pii_pipeline.scoring import ConfidenceCalibrator

logger = logging.getLogger(__name__)


def run_system(name: str, system, records: list[dict[str, Any]]) -> tuple[list[dict], float]:
    docs: list[dict[str, Any]] = []
    started = time.perf_counter()
    for i, rec in enumerate(records):
        if i and i % 200 == 0:
            logger.info("  %s: %d/%d", name, i, len(records))
        pred = to_spans(system(rec["text"], rec["lang"]))
        docs.append({"text": rec["text"], "gold": rec["entities"], "pred": pred})
    return docs, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--synthetic", type=Path, default=Path("data/synthetic/documents.jsonl"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/distilbert-pii-ner/final"))
    parser.add_argument("--calibrator", type=Path, default=Path("models/calibrator.json"))
    parser.add_argument("--out", type=Path, default=Path("results/metrics/evaluation.json"))
    parser.add_argument("--n-ai4privacy", type=int, default=800)
    parser.add_argument("--n-synthetic", type=int, default=300)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    calibrator = ConfidenceCalibrator.load(args.calibrator)
    systems = build_systems(args.model_dir, calibrator)
    logger.info("systems: %s", list(systems))

    test_sets = {
        "ai4privacy": read_jsonl(args.data_dir / "test.jsonl")[: args.n_ai4privacy],
        "synthetic_documents": read_jsonl(args.synthetic)[: args.n_synthetic],
    }

    results: dict[str, Any] = {
        "note": (
            "The ai4privacy corpus contains no card or bank-account numbers, so "
            f"{sorted(TYPES_WITHOUT_CORPUS_SUPPORT)} are unreachable for the "
            "fine-tuned model by construction and are covered by the checksum-backed "
            "rule engine inside the ensemble."
        ),
        "types_without_corpus_support": sorted(TYPES_WITHOUT_CORPUS_SUPPORT),
        "test_sets": {},
    }

    for ts_name, records in test_sets.items():
        logger.info("=== test set %s (%d docs) ===", ts_name, len(records))
        results["test_sets"][ts_name] = {"n_documents": len(records), "systems": {}}
        for sys_name, system in systems.items():
            docs, elapsed = run_system(sys_name, system, records)
            entry = {
                "exact": evaluate(docs, mode="exact"),
                "partial": evaluate(docs, mode="partial", iou_threshold=0.5),
                "seconds": round(elapsed, 2),
                "docs_per_second": round(len(records) / elapsed, 2) if elapsed else 0.0,
            }
            results["test_sets"][ts_name]["systems"][sys_name] = entry
            ex, pa = entry["exact"], entry["partial"]
            logger.info(
                "  %-24s exactF1=%.4f partialF1=%.4f leak=%.4f over-red=%.5f (%.1fs)",
                sys_name, ex["micro"]["f1"], pa["micro"]["f1"],
                pa["privacy"]["entity_leakage_rate"],
                pa["privacy"]["over_redaction_rate"], elapsed,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
