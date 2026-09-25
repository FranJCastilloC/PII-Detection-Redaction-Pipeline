"""Fine-tune the PII token-classification model on the processed splits."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pii_pipeline.data.build_dataset import read_jsonl
from pii_pipeline.training import BASE_MODEL, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/distilbert-pii-ner"))
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Cap on training records. The corpus is highly templated, so a "
        "fraction of it converges just as well and keeps the run inside the "
        "memory budget of a 16 GB laptop.",
    )
    parser.add_argument("--limit-val", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_records = read_jsonl(args.data_dir / "train.jsonl")
    val_records = read_jsonl(args.data_dir / "validation.jsonl")
    if args.limit_train:
        train_records = train_records[: args.limit_train]
    if args.limit_val:
        val_records = val_records[: args.limit_val]

    report = train(
        train_records=train_records,
        val_records=val_records,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
    )
    final = report["final_validation"]
    print(
        f"F1 validacion: {final.get('eval_f1', 0):.4f}  "
        f"P: {final.get('eval_precision', 0):.4f}  R: {final.get('eval_recall', 0):.4f}"
    )


if __name__ == "__main__":
    main()
