"""Detector protocol and span-algebra helpers shared by every detector."""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from ..entities import PIIEntity, PIIType

#: Types whose value may legitimately wrap across lines in a real document.
#: Everything else is truncated at the first newline: statistical NER regularly
#: runs a PERSON span straight through a line break and into the next field's
#: label, and redacting that deletes the label along with the name.
MULTILINE_TYPES: frozenset[str] = frozenset({PIIType.ADDRESS.value})

#: Characters stripped from the edges of a detected span. Detectors routinely
#: swallow a trailing period or a wrapping parenthesis; redacting those would
#: silently corrupt the surrounding sentence.
_TRIM_CHARS = " \t\n\r.,;:!?\"'()[]{}<>«»¡¿"


@runtime_checkable
class Detector(Protocol):
    """Anything that turns text into PII spans."""

    name: str

    def detect(self, text: str, lang: str = "es") -> list[PIIEntity]:
        ...


def trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Shrink ``[start, end)`` so it excludes leading/trailing noise.

    Returns ``None`` when nothing but noise is left.
    """
    while start < end and text[start] in _TRIM_CHARS:
        start += 1
    while end > start and text[end - 1] in _TRIM_CHARS:
        end -= 1
    return (start, end) if end > start else None


def normalize_entities(text: str, entities: Iterable[PIIEntity]) -> list[PIIEntity]:
    """Trim edges and drop spans that fall outside the text or collapse to empty."""
    out: list[PIIEntity] = []
    n = len(text)
    for ent in entities:
        start, end = max(0, ent.start), min(n, ent.end)
        if end <= start:
            continue
        if ent.type not in MULTILINE_TYPES:
            newline = text.find("\n", start, end)
            if newline != -1:
                end = newline
        trimmed = trim_span(text, start, end)
        if trimmed is None:
            continue
        start, end = trimmed
        ent.start, ent.end = start, end
        ent.text = text[start:end]
        out.append(ent)
    return out


def checksum_rank(ent: PIIEntity) -> int:
    """+1 when a checksum verified the span, -1 when one failed, 0 when absent.

    This outranks every other criterion: a mod-97-verified IBAN must never lose
    an overlap to a digit run that merely *looks* like a card number and failed
    Luhn. Getting this order wrong silently leaks the stronger identifier.
    """
    passed = ent.metadata.get("checksum_passed")
    if passed is True:
        return 1
    if passed is False:
        return -1
    return 0


def _rank_key(ent: PIIEntity) -> tuple[int, float, int, int]:
    """Sort key deciding who wins an overlap: checksum, score, type, length."""
    return (checksum_rank(ent), round(ent.score, 3), ent.priority, ent.length)


def _dominates(a: PIIEntity, b: PIIEntity) -> bool:
    """True when ``a`` should win an overlap conflict against ``b``."""
    return _rank_key(a) > _rank_key(b)


def resolve_overlaps(entities: Iterable[PIIEntity]) -> list[PIIEntity]:
    """Greedily keep the strongest entity out of every overlapping cluster.

    Redaction rewrites the string, so two overlapping spans cannot both be
    applied. Losing candidates are not thrown away silently: the winner records
    them under ``metadata['suppressed']`` so the review queue can flag the
    disagreement to a human.
    """
    ranked = sorted(entities, key=lambda e: (_neg(_rank_key(e)), e.start))
    kept: list[PIIEntity] = []
    for cand in ranked:
        clash = next((k for k in kept if k.overlaps(cand)), None)
        if clash is None:
            kept.append(cand)
            continue
        if _dominates(cand, clash):
            # Should not happen given the sort order, but keeps the invariant
            # explicit rather than relying on it.
            kept.remove(clash)
            cand.metadata.setdefault("suppressed", []).append(_brief(clash))
            kept.append(cand)
        else:
            clash.metadata.setdefault("suppressed", []).append(_brief(cand))
    return sorted(kept, key=lambda e: e.start)


def _neg(key: tuple[int, float, int, int]) -> tuple[float, ...]:
    """Negate a rank key so ``sorted`` yields strongest-first."""
    return tuple(-v for v in key)


def _brief(ent: PIIEntity) -> dict[str, object]:
    return {
        "type": ent.type,
        "start": ent.start,
        "end": ent.end,
        "score": round(ent.score, 4),
        "source": ent.source,
    }
