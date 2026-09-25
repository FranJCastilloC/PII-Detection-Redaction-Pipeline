"""Fine-tuning of a multilingual transformer for PII token classification.

Character offsets from the annotated JSONL are aligned to the model's own
subword tokenisation at encode time, so the project never depends on a
pre-tokenised corpus. Long documents are split with a sliding window
(``return_overflowing_tokens`` + ``stride``) instead of being truncated, which
matters because business documents routinely exceed 256 tokens and the PII in a
contract is often at the end.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .entities import ID2LABEL, LABEL2ID, MODEL_BIO_LABELS

logger = logging.getLogger(__name__)

BASE_MODEL = "distilbert-base-multilingual-cased"
MAX_LENGTH = 256
STRIDE = 64
#: Label id used by the loss to ignore a position (special tokens, continuation
#: subwords of an already-labelled word, and the overlap region of a window).
IGNORE_INDEX = -100


def align_labels(
    offsets: list[tuple[int, int]],
    sequence_ids: list[int | None],
    entities: list[dict[str, Any]],
) -> list[int]:
    """Project character-level entity spans onto subword tokens as BIO ids."""
    labels: list[int] = []
    for (start, end), seq_id in zip(offsets, sequence_ids):
        if seq_id is None or end <= start:
            labels.append(IGNORE_INDEX)
            continue
        tag = "O"
        for ent in entities:
            if start >= ent["end"] or end <= ent["start"]:
                continue
            # A token belongs to the entity it overlaps; it opens the span only
            # when it is the first token to touch it.
            prefix = "B" if start <= ent["start"] else "I"
            candidate = f"{prefix}-{ent['type']}"
            if candidate in LABEL2ID:
                tag = candidate
            break
        labels.append(LABEL2ID.get(tag, LABEL2ID["O"]))
    return labels


def iter_features(
    records: list[dict[str, Any]],
    tokenizer,
    max_length: int = MAX_LENGTH,
    stride: int = STRIDE,
):
    """Yield windowed features one at a time.

    Materialising all windows in a Python list before handing them to Arrow was
    enough to push a 16 GB machine into swap, so encoding is streamed instead.
    """
    for rec in records:
        encoding = tokenizer(
            rec["text"],
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        entities = sorted(rec.get("entities", []), key=lambda e: e["start"])
        for window in range(len(encoding["input_ids"])):
            offsets = encoding["offset_mapping"][window]
            labels = align_labels(
                offsets, encoding.encodings[window].sequence_ids, entities
            )
            yield {
                "input_ids": encoding["input_ids"][window],
                "attention_mask": encoding["attention_mask"][window],
                "labels": labels,
            }


def encode_records(
    records: list[dict[str, Any]],
    tokenizer,
    max_length: int = MAX_LENGTH,
    stride: int = STRIDE,
) -> list[dict[str, Any]]:
    """Eager variant of :func:`iter_features`, used by tests and notebooks."""
    features: list[dict[str, Any]] = []
    for rec in records:
        encoding = tokenizer(
            rec["text"],
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        entities = sorted(rec.get("entities", []), key=lambda e: e["start"])
        for window in range(len(encoding["input_ids"])):
            offsets = encoding["offset_mapping"][window]
            labels = align_labels(
                offsets, encoding.encodings[window].sequence_ids, entities
            )
            features.append(
                {
                    "input_ids": encoding["input_ids"][window],
                    "attention_mask": encoding["attention_mask"][window],
                    "labels": labels,
                }
            )
    return features


def build_compute_metrics():
    """Return a ``compute_metrics`` closure scoring entity-level seqeval F1."""
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

    def compute_metrics(eval_pred):
        # ``preprocess_logits_for_metrics`` already reduced logits to label ids:
        # keeping the full [n_windows, seq_len, n_labels] float tensor around was
        # a ~84x memory multiplier on the evaluation set.
        preds, gold = eval_pred
        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)
        pred_tags, gold_tags = [], []
        for pred_row, gold_row in zip(preds, gold):
            keep = gold_row != IGNORE_INDEX
            pred_tags.append([ID2LABEL[int(p)] for p in pred_row[keep]])
            gold_tags.append([ID2LABEL[int(g)] for g in gold_row[keep]])
        return {
            "precision": precision_score(gold_tags, pred_tags),
            "recall": recall_score(gold_tags, pred_tags),
            "f1": f1_score(gold_tags, pred_tags),
            "report": classification_report(gold_tags, pred_tags, output_dict=True, zero_division=0),
        }

    return compute_metrics


def train(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    output_dir: Path,
    base_model: str = BASE_MODEL,
    epochs: float = 3.0,
    batch_size: int = 32,
    learning_rate: float = 5e-5,
    max_length: int = MAX_LENGTH,
    seed: int = 13,
) -> dict[str, Any]:
    from datasets import Dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    logger.info("encoding %d train / %d val records", len(train_records), len(val_records))
    train_ds = Dataset.from_generator(
        iter_features,
        gen_kwargs={"records": train_records, "tokenizer": tokenizer, "max_length": max_length},
    )
    val_ds = Dataset.from_generator(
        iter_features,
        gen_kwargs={"records": val_records, "tokenizer": tokenizer, "max_length": max_length},
    )
    logger.info("windows: %d train / %d val", len(train_ds), len(val_ds))

    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(MODEL_BIO_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    steps_per_epoch = max(1, len(train_ds) // batch_size)
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_steps=int(0.06 * steps_per_epoch * epochs),
        logging_steps=100,
        seed=seed,
        report_to=[],
        dataloader_num_workers=0,
        eval_accumulation_steps=16,
    )

    def preprocess_logits_for_metrics(logits, labels):
        return logits.argmax(dim=-1)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=build_compute_metrics(),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metrics = trainer.evaluate()
    history = [h for h in trainer.state.log_history if "loss" in h or "eval_loss" in h]
    payload = {
        "base_model": base_model,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "final_validation": {k: v for k, v in metrics.items() if k != "eval_report"},
        "per_type_validation": metrics.get("eval_report", {}),
        "history": history,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    return payload
