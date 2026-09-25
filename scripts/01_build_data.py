"""Download the corpus, build the annotated splits and generate synthetic docs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from pii_pipeline.data.build_dataset import build
from pii_pipeline.data.download_ai4privacy import download
from pii_pipeline.data.synth_documents import generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--synthetic", type=Path, default=Path("data/synthetic/documents.jsonl"))
    parser.add_argument("--train-size", type=int, default=40000)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=6000)
    parser.add_argument("--synthetic-docs", type=int, default=300)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.skip_download:
        download(args.raw_dir)
    report = build(args.raw_dir, args.out_dir, args.train_size, args.val_size, args.test_size)

    docs = generate(args.synthetic_docs)
    args.synthetic.parent.mkdir(parents=True, exist_ok=True)
    with args.synthetic.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(json.dumps(report["splits"], ensure_ascii=False, indent=2))
    print(f"{len(docs)} synthetic documents -> {args.synthetic}")


if __name__ == "__main__":
    main()
