"""End-to-end orchestration: detect -> fuse -> calibrate -> route -> redact."""

from __future__ import annotations

import logging
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .detectors.presidio_detector import PresidioDetector
from .detectors.regex_rules import RegexRuleDetector
from .detectors.transformer_detector import DEFAULT_MODEL_DIR, TransformerDetector
from .entities import PIIEntity
from .redaction import RedactionMode, redact, verify_no_leakage
from .review_queue import (
    Decision,
    ReviewItem,
    ReviewPolicy,
    ReviewQueue,
    build_context,
)
from .scoring import ConfidenceCalibrator, fuse

logger = logging.getLogger(__name__)

ENGINE_BASELINE = ("rules", "presidio")
ENGINE_FULL = ("rules", "presidio", "model")

_ES_MARKERS = re.compile(
    r"\b(el|la|los|las|de|del|que|para|con|por|una|nombre|tel[eé]fono|direcci[oó]n|"
    r"fecha|cliente|cuenta|correo|documento|factura|contrato)\b",
    re.IGNORECASE,
)
_EN_MARKERS = re.compile(
    r"\b(the|and|for|with|from|your|name|phone|address|date|customer|account|email|"
    r"invoice|agreement|please)\b",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    """Cheap ES/EN discriminator: accented characters plus stop-word counts."""
    es = len(_ES_MARKERS.findall(text)) + 2 * len(re.findall(r"[áéíóúñ¿¡]", text, re.I))
    en = len(_EN_MARKERS.findall(text))
    return "es" if es >= en else "en"


@dataclass
class PipelineResult:
    document_id: str
    language: str
    original_text: str
    redacted_text: str
    entities: list[PIIEntity] = field(default_factory=list)
    decisions: dict[str, str] = field(default_factory=dict)
    review_items: list[ReviewItem] = field(default_factory=list)
    leaks: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def entities_by_decision(self, decision: Decision | str) -> list[PIIEntity]:
        value = Decision(decision).value
        return [e for e in self.entities if self.decisions.get(_key(e)) == value]

    def audit_log(self) -> dict[str, Any]:
        """Serialisable record of everything the pipeline did to this document."""
        return {
            "document_id": self.document_id,
            "language": self.language,
            "stats": self.stats,
            "leaks": self.leaks,
            "entities": [
                {**e.to_dict(), "decision": self.decisions.get(_key(e))}
                for e in self.entities
            ],
            "review_items": [item.to_dict() for item in self.review_items],
        }


def _unify_repeated_values(entities: list[PIIEntity]) -> list[PIIEntity]:
    """Give every mention of the same value in a document the same confidence.

    Without this, two occurrences of one value can land on opposite sides of a
    threshold -- observed in practice: the same string scored 0.51 in the header
    and 0.48 in the body, so the first was redacted and the second was left in
    the released document. A value is either sensitive in this document or it is
    not; the decision cannot depend on which mention you happen to look at.
    """
    best: dict[tuple[str, str], float] = {}
    for ent in entities:
        key = (ent.text.strip().casefold(), ent.type)
        best[key] = max(best.get(key, 0.0), ent.score)
    for ent in entities:
        key = (ent.text.strip().casefold(), ent.type)
        if best[key] > ent.score:
            ent.metadata["score_before_repeat_propagation"] = round(ent.score, 4)
            ent.score = best[key]
    return entities


def sweep_repeated_occurrences(
    text: str, entities: list[PIIEntity], min_length: int = 4
) -> list[PIIEntity]:
    """Redact every other exact occurrence of a value already judged sensitive.

    Detectors work span by span and routinely catch a value in the header while
    missing the identical string in the body. Once the pipeline has decided a
    value is sensitive in this document, leaving another copy of it visible is
    indefensible -- so the decision is applied to the whole document.

    Matches are exact and word-bounded, and short values are skipped, so this
    cannot fire inside an unrelated word.
    """
    occupied = [(e.start, e.end) for e in entities]
    extra: list[PIIEntity] = []
    for ent in entities:
        value = ent.text.strip()
        if len(value) < min_length:
            continue
        for match in re.finditer(re.escape(value), text):
            start, end = match.span()
            if any(start < b and a < end for a, b in occupied):
                continue
            if start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
                continue
            if end < len(text) and (text[end].isalnum() or text[end] == "_"):
                continue
            occupied.append((start, end))
            extra.append(
                PIIEntity(
                    text=text[start:end],
                    type=ent.type,
                    start=start,
                    end=end,
                    score=ent.score,
                    source=ent.source,
                    metadata={**ent.metadata, "found_by": "repeat_sweep"},
                )
            )
    return sorted(entities + extra, key=lambda e: e.start)


def _key(ent: PIIEntity) -> str:
    return f"{ent.start}:{ent.end}:{ent.type}"


class PIIPipeline:
    """Detect, score, route and redact PII in a business document."""

    def __init__(
        self,
        engines: Sequence[str] = ENGINE_FULL,
        model_dir: Path | str = DEFAULT_MODEL_DIR,
        policy: ReviewPolicy | None = None,
        calibrator: ConfidenceCalibrator | None = None,
        queue: ReviewQueue | None = None,
        max_length: int = 192,
    ):
        self.policy = policy or ReviewPolicy()
        self.calibrator = calibrator or ConfidenceCalibrator()
        self.queue = queue
        self.detectors: dict[str, Any] = {}

        if "rules" in engines:
            self.detectors["rules"] = RegexRuleDetector()
        if "presidio" in engines:
            self.detectors["presidio"] = PresidioDetector()
        if "model" in engines:
            detector = TransformerDetector(model_dir, max_length=max_length)
            if detector.is_available:
                self.detectors["model"] = detector
            else:
                logger.warning(
                    "no fine-tuned model at %s; continuing without it", model_dir
                )

    @property
    def engines(self) -> list[str]:
        return list(self.detectors)

    # -- main --------------------------------------------------------------
    def process(
        self,
        text: str,
        lang: str | None = None,
        mode: RedactionMode | str = RedactionMode.MASK,
        document_id: str | None = None,
        enqueue: bool = True,
    ) -> PipelineResult:
        document_id = document_id or f"doc-{uuid.uuid4().hex[:10]}"
        lang = lang or detect_language(text)

        raw: list[PIIEntity] = []
        per_engine: dict[str, int] = {}
        for name, detector in self.detectors.items():
            found = detector.detect(text, lang)
            per_engine[name] = len(found)
            raw.extend(found)

        entities = _unify_repeated_values(self.calibrator.apply(fuse(raw)))

        decisions: dict[str, str] = {}
        review_items: list[ReviewItem] = []
        to_redact: list[PIIEntity] = []
        for ent in entities:
            decision, reasons = self.policy.decide(ent)
            decisions[_key(ent)] = decision.value
            ent.metadata["decision"] = decision.value
            ent.metadata["decision_reasons"] = reasons
            if decision is Decision.DISCARD:
                continue
            # Anything not discarded is redacted, including the uncertain cases:
            # a reviewer can always put a false positive back, but a leak that
            # already left the building cannot be recalled.
            to_redact.append(ent)
            if decision is Decision.REVIEW:
                review_items.append(
                    ReviewItem(
                        document_id=document_id,
                        entity_type=ent.type,
                        text=ent.text,
                        start=ent.start,
                        end=ent.end,
                        score=round(ent.score, 4),
                        reasons=reasons,
                        context=build_context(text, ent.start, ent.end),
                        source=ent.source,
                    )
                )

        to_redact = sweep_repeated_occurrences(text, to_redact)
        output = redact(text, to_redact, mode=mode, lang=lang)
        leaks = verify_no_leakage(output)
        if leaks:
            logger.error("%s: %d values survived redaction", document_id, len(leaks))

        if enqueue and self.queue is not None and review_items:
            self.queue.enqueue(review_items)

        counts = Counter(decisions.values())
        stats = {
            "n_entities": len(entities),
            "n_auto_redacted": counts.get(Decision.AUTO_REDACT.value, 0),
            "n_review": counts.get(Decision.REVIEW.value, 0),
            "n_discarded": counts.get(Decision.DISCARD.value, 0),
            "review_rate": round(
                counts.get(Decision.REVIEW.value, 0) / len(entities), 4
            ) if entities else 0.0,
            "by_type": dict(Counter(e.type for e in entities).most_common()),
            "detections_per_engine": per_engine,
            "engines": self.engines,
            "redaction_mode": RedactionMode(mode).value,
        }

        return PipelineResult(
            document_id=document_id,
            language=lang,
            original_text=text,
            redacted_text=output.redacted_text,
            entities=entities,
            decisions=decisions,
            review_items=review_items,
            leaks=leaks,
            stats=stats,
        )
