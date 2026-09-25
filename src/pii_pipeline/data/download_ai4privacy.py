"""Download the ai4privacy PII corpus and keep only the English/Spanish slice.

Why this corpus: it is the only large public dataset that annotates the
*structured* identifiers a business-document workflow actually has to remove
(cards, IBANs, tax IDs, dates of birth) rather than only the PER/LOC/ORG classes
of classic NER benchmarks. It is fully synthetic, so no real person's data is
ever pulled into the project.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_ID = "ai4privacy/pii-masking-300k"
#: The corpus spells languages out in full.
KEEP_LANGUAGES = {"English", "Spanish"}
KEEP_COLUMNS = ["source_text", "privacy_mask", "language"]


def download(out_dir: Path, dataset_id: str = DATASET_ID) -> dict[str, Path]:
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("loading %s from the Hugging Face hub", dataset_id)
    dsets = load_dataset(dataset_id)

    written: dict[str, Path] = {}
    for split, dset in dsets.items():
        before = len(dset)
        dset = dset.filter(lambda row: row["language"] in KEEP_LANGUAGES, num_proc=4)
        dset = dset.select_columns([c for c in KEEP_COLUMNS if c in dset.column_names])
        path = out_dir / f"ai4privacy_{split}.parquet"
        dset.to_parquet(str(path))
        written[split] = path
        logger.info("%s: %d -> %d rows (EN/ES) -> %s", split, before, len(dset), path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset-id", default=DATASET_ID)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download(args.out_dir, args.dataset_id)


if __name__ == "__main__":
    main()
