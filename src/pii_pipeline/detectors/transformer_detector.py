"""Inference for the fine-tuned multilingual token-classification model.

Two details matter more than the model weights themselves:

* **Sliding windows.** Business documents are long. Truncating at 256 tokens
  would silently drop the PII in the second half of a contract, and a redaction
  system that silently drops PII is worse than no system at all.
* **Per-span confidence.** The downstream review queue needs a number it can
  threshold, so each span carries the geometric mean of its token probabilities
  plus the top1-top2 margin, not just a hard label.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..entities import PIIEntity
from .base import normalize_entities, resolve_overlaps

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path("models/distilbert-pii-ner/final")

#: Hugging Face repo used when no local checkpoint is present. The trained
#: weights are ~500 MB and stay out of git; the hosted demo pulls them from the
#: Hub instead, so the deployed app runs the same ensemble as a local checkout
#: rather than silently degrading to the rule baseline.
DEFAULT_HUB_REPO = os.environ.get("PII_MODEL_REPO", "")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class TransformerDetector:
    """Fine-tuned transformer exposed through the ``Detector`` protocol."""

    name = "model"

    def __init__(
        self,
        model_dir: Path | str = DEFAULT_MODEL_DIR,
        max_length: int = 192,
        stride: int = 64,
        device: str | None = None,
        min_score: float = 0.30,
        hub_repo: str = DEFAULT_HUB_REPO,
    ):
        self.model_dir = Path(model_dir)
        self.hub_repo = hub_repo
        self.max_length = max_length
        self.stride = stride
        self.min_score = min_score
        self._device = device
        self._model = None
        self._tokenizer = None

    # -- lazy init ---------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        if self._device is None:
            self._device = (
                "mps"
                if torch.backends.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        source = self.source
        if source is None:
            raise FileNotFoundError(
                f"no model at {self.model_dir} and no Hugging Face repo configured "
                "(set PII_MODEL_REPO)"
            )
        self._tokenizer = AutoTokenizer.from_pretrained(source)
        self._model = AutoModelForTokenClassification.from_pretrained(source)
        self._model.to(self._device).eval()
        self._id2label = self._model.config.id2label
        logger.info("loaded %s on %s", source, self._device)

    @property
    def source(self) -> str | None:
        """Where the weights come from: the local checkpoint, else the Hub repo."""
        if (self.model_dir / "config.json").exists():
            return str(self.model_dir)
        return self.hub_repo or None

    @property
    def is_available(self) -> bool:
        return self.source is not None

    # -- decoding ----------------------------------------------------------
    def _spans_from_window(
        self,
        text: str,
        offsets: list[tuple[int, int]],
        seq_ids: list[int | None],
        probs: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Turn one window's token probabilities into BIO-decoded spans."""
        spans: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        for idx, ((start, end), seq_id) in enumerate(zip(offsets, seq_ids)):
            if seq_id is None or end <= start:
                continue
            row = probs[idx]
            best = int(row.argmax())
            label = self._id2label[best]
            confidence = float(row[best])
            ordered = np.partition(row, -2)
            margin = float(confidence - ordered[-2])

            if label == "O":
                if current:
                    spans.append(current)
                    current = None
                continue

            prefix, _, pii_type = label.partition("-")
            if prefix == "B" or current is None or current["type"] != pii_type:
                if current:
                    spans.append(current)
                current = {
                    "type": pii_type,
                    "start": start,
                    "end": end,
                    "probs": [confidence],
                    "margins": [margin],
                }
            else:
                current["end"] = end
                current["probs"].append(confidence)
                current["margins"].append(margin)

        if current:
            spans.append(current)
        return spans

    # -- main --------------------------------------------------------------
    def detect(self, text: str, lang: str = "es") -> list[PIIEntity]:
        if not text.strip():
            return []
        self._ensure_loaded()
        import torch

        encoding = self._tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._model(
                input_ids=encoding["input_ids"].to(self._device),
                attention_mask=encoding["attention_mask"].to(self._device),
            ).logits.cpu().numpy()

        probs = _softmax(logits)
        # Windows overlap by `stride`, so the same span can be decoded twice.
        # Keyed dedup keeps the most confident reading of each span.
        best_by_span: dict[tuple[int, int, str], dict[str, Any]] = {}
        for window in range(len(encoding["input_ids"])):
            offsets = [tuple(o) for o in encoding["offset_mapping"][window].tolist()]
            seq_ids = encoding.encodings[window].sequence_ids
            for span in self._spans_from_window(text, offsets, seq_ids, probs[window]):
                key = (span["start"], span["end"], span["type"])
                prev = best_by_span.get(key)
                if prev is None or np.mean(span["probs"]) > np.mean(prev["probs"]):
                    best_by_span[key] = span

        entities: list[PIIEntity] = []
        for span in best_by_span.values():
            # Geometric mean: one weak token drags the whole span down, which is
            # the behaviour we want when deciding whether a human should look.
            score = float(np.exp(np.mean(np.log(np.clip(span["probs"], 1e-9, 1.0)))))
            if score < self.min_score:
                continue
            entities.append(
                PIIEntity(
                    text=text[span["start"] : span["end"]],
                    type=span["type"],
                    start=span["start"],
                    end=span["end"],
                    score=score,
                    source=self.name,
                    metadata={
                        "n_tokens": len(span["probs"]),
                        "min_token_prob": round(float(min(span["probs"])), 4),
                        "mean_margin": round(float(np.mean(span["margins"])), 4),
                    },
                )
            )
        return resolve_overlaps(normalize_entities(text, entities))
