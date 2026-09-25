"""Turn the raw ai4privacy parquet into the project's annotated JSONL splits.

Output record::

    {"id": ..., "lang": "es", "text": "...",
     "entities": [{"start": 12, "end": 21, "type": "PERSON", "text": "Ana Ruiz"}]}

Character offsets are kept as the source of truth rather than the corpus's
pre-computed mBERT BIO tags, so the tokenisation stays owned by this project and
the same annotations can score the rule baseline, Presidio and the model.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from ..entities import (
    INTENTIONALLY_DROPPED,
    TYPES_WITHOUT_CORPUS_SUPPORT,
    map_source_label,
)

logger = logging.getLogger(__name__)

LANG_CODE = {"English": "en", "Spanish": "es"}


def _iter_rows(parquet_path: Path) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    for batch in table.to_batches(max_chunksize=2048):
        yield from batch.to_pylist()


def _clean_entities(
    text: str, mask: list[dict[str, Any]], dropped: Counter
) -> list[dict[str, Any]]:
    """Map raw labels onto the taxonomy, verify offsets and drop overlaps."""
    out: list[dict[str, Any]] = []
    for item in mask or []:
        raw_label = str(item.get("label", "")).upper()
        mapped = map_source_label(raw_label)
        if mapped is None:
            dropped[raw_label] += 1
            continue
        start, end = int(item["start"]), int(item["end"])
        if not (0 <= start < end <= len(text)):
            dropped[f"__bad_offset__{raw_label}"] += 1
            continue
        # The corpus occasionally stores a value that no longer lines up with the
        # text; trusting it would train the model on noise.
        if item.get("value") and text[start:end] != item["value"]:
            dropped[f"__offset_mismatch__{raw_label}"] += 1
            continue
        out.append({"start": start, "end": end, "type": mapped, "text": text[start:end]})

    out.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
    deduped: list[dict[str, Any]] = []
    for ent in out:
        if deduped and ent["start"] < deduped[-1]["end"]:
            continue  # overlapping annotation: keep the first, longest one
        deduped.append(ent)
    return deduped


def build_records(parquet_path: Path, dropped: Counter) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(_iter_rows(parquet_path)):
        text = row.get("source_text") or ""
        if not text.strip():
            continue
        lang = LANG_CODE.get(row.get("language", ""), "en")
        entities = _clean_entities(text, row.get("privacy_mask"), dropped)
        records.append(
            {
                "id": f"{parquet_path.stem}-{idx}",
                "lang": lang,
                "text": text,
                "entities": entities,
            }
        )
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _balance_by_language(
    records: list[dict[str, Any]], limit: int | None, rng: random.Random
) -> list[dict[str, Any]]:
    """Sample up to ``limit`` records with an even split across languages."""
    if limit is None or limit >= len(records):
        rng.shuffle(records)
        return records
    by_lang: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_lang.setdefault(rec["lang"], []).append(rec)
    per_lang = max(1, limit // max(1, len(by_lang)))
    picked: list[dict[str, Any]] = []
    for lang, group in by_lang.items():
        rng.shuffle(group)
        picked.extend(group[:per_lang])
    rng.shuffle(picked)
    return picked[:limit]


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    langs, types = Counter(), Counter()
    empty = 0
    for rec in records:
        langs[rec["lang"]] += 1
        if not rec["entities"]:
            empty += 1
        for ent in rec["entities"]:
            types[ent["type"]] += 1
    return {
        "n_documents": len(records),
        "n_entities": sum(types.values()),
        "documents_without_pii": empty,
        "by_language": dict(langs),
        "by_type": dict(types.most_common()),
    }


def build(
    raw_dir: Path,
    out_dir: Path,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int = 13,
) -> dict[str, Any]:
    rng = random.Random(seed)
    dropped: Counter = Counter()

    train_pool = build_records(raw_dir / "ai4privacy_train.parquet", dropped)
    test_pool = build_records(raw_dir / "ai4privacy_validation.parquet", dropped)

    # The corpus ships train/validation; we carve our validation set out of the
    # training pool and reserve the official validation split as an untouched
    # test set, so no tuning decision ever sees the numbers we report.
    train_pool = _balance_by_language(train_pool, train_size + val_size, rng)
    train_records = train_pool[:train_size]
    val_records = train_pool[train_size : train_size + val_size]
    test_records = _balance_by_language(test_pool, test_size, rng)

    splits = {"train": train_records, "validation": val_records, "test": test_records}
    for name, recs in splits.items():
        write_jsonl(recs, out_dir / f"{name}.jsonl")

    report = {
        "source_dataset": "ai4privacy/pii-masking-300k",
        "seed": seed,
        "splits": {name: summarise(recs) for name, recs in splits.items()},
        "dropped_source_labels": {
            "counts": dict(dropped.most_common()),
            "note": (
                "Labels mapped to None are demographic/descriptive attributes kept "
                "out of the 12-class taxonomy on purpose; __bad_offset__ and "
                "__offset_mismatch__ are corpus annotations whose character offsets "
                "did not line up with the text and were discarded."
            ),
            "intentionally_dropped_catalogue": dict(INTENTIONALLY_DROPPED),
            "types_without_corpus_support": sorted(TYPES_WITHOUT_CORPUS_SUPPORT),
        },
    }
    (out_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-size", type=int, default=40000)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = build(
        args.raw_dir, args.out_dir, args.train_size, args.val_size, args.test_size, args.seed
    )
    print(json.dumps(report["splits"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
