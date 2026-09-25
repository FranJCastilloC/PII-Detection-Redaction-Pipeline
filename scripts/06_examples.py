"""Run the full pipeline over sample documents and save input/output pairs.

Produces the artefacts a reader looks at first: one redacted example per document
type, the audit JSON behind it, and the aggregate routing statistics that show
how much work the review queue actually receives.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pii_pipeline.data.build_dataset import read_jsonl
from pii_pipeline.pipeline import PIIPipeline
from pii_pipeline.review_queue import ReviewPolicy, ReviewQueue
from pii_pipeline.scoring import ConfidenceCalibrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, default=Path("data/synthetic/documents.jsonl"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/distilbert-pii-ner/final"))
    parser.add_argument("--calibrator", type=Path, default=Path("models/calibrator.json"))
    parser.add_argument("--examples-dir", type=Path, default=Path("results/examples"))
    parser.add_argument("--queue-db", type=Path, default=Path("data/review_queue.sqlite"))
    parser.add_argument("--queue-stats", type=Path, default=Path("results/metrics/queue_stats.json"))
    parser.add_argument("-n", "--num-docs", type=int, default=120)
    args = parser.parse_args()

    args.examples_dir.mkdir(parents=True, exist_ok=True)
    queue = ReviewQueue(args.queue_db)
    queue.clear()

    pipeline = PIIPipeline(
        model_dir=args.model_dir,
        calibrator=ConfidenceCalibrator.load(args.calibrator),
        policy=ReviewPolicy(),
        queue=queue,
    )

    docs = read_jsonl(args.synthetic)[: args.num_docs]
    decisions: Counter = Counter()
    leaks_total = 0
    saved: set[tuple[str, str]] = set()

    for doc in docs:
        result = pipeline.process(
            doc["text"], lang=doc["lang"], mode="mask", document_id=doc["id"]
        )
        decisions["auto_redact"] += result.stats["n_auto_redacted"]
        decisions["review"] += result.stats["n_review"]
        decisions["discard"] += result.stats["n_discarded"]
        leaks_total += len(result.leaks)

        # One saved example per (document type, language) pair.
        key = (doc["doc_type"], doc["lang"])
        if key not in saved:
            saved.add(key)
            stem = f"{doc['doc_type']}_{doc['lang']}"
            (args.examples_dir / f"{stem}.input.txt").write_text(
                result.original_text, encoding="utf-8"
            )
            (args.examples_dir / f"{stem}.redacted.txt").write_text(
                result.redacted_text, encoding="utf-8"
            )
            (args.examples_dir / f"{stem}.audit.json").write_text(
                json.dumps(result.audit_log(), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

    total = sum(decisions.values())
    stats = {
        "n_documents": len(docs),
        "decisions": dict(decisions),
        "review_rate": round(decisions["review"] / total, 4) if total else 0.0,
        "auto_rate": round(decisions["auto_redact"] / total, 4) if total else 0.0,
        "verbatim_leaks_after_redaction": leaks_total,
        "queue_status": queue.stats(),
    }
    args.queue_stats.parent.mkdir(parents=True, exist_ok=True)
    args.queue_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"ejemplos -> {args.examples_dir}")


if __name__ == "__main__":
    main()
