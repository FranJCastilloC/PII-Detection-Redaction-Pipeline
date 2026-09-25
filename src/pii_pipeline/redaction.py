"""Applying redaction to text without corrupting character offsets.

Three strategies, because "redact everything" is rarely what a business wants:

``mask``
    Replace with a typed marker (``[NOMBRE]``). Maximum privacy, destroys the
    ability to tell two different people apart in the same document.
``partial``
    Keep the last few characters of structured identifiers, the way a bank shows
    a card as ``**** 6467``. Enough for a human to reconcile a record, not enough
    to reuse the identifier.
``pseudonymize``
    Replace with a deterministic HMAC-derived tag (``[NOMBRE:a3f9c2]``). The same
    person maps to the same tag across documents, so analytics still work, while
    reversing it requires the secret key.
"""

from __future__ import annotations

import hmac
import re
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from typing import Any, Iterable

from .entities import PIIEntity, marker_for

#: Types where keeping a short suffix is meaningful and safe enough.
_PARTIAL_TAIL = {
    "CREDIT_CARD": 4,
    "BANK_ACCOUNT": 4,
    "PHONE": 3,
    "GOV_ID": 3,
}


class RedactionMode(str, Enum):
    MASK = "mask"
    PARTIAL = "partial"
    PSEUDONYMIZE = "pseudonymize"


@dataclass
class Replacement:
    """One applied substitution, in the coordinates of both texts."""

    original: str
    replacement: str
    type: str
    source_start: int
    source_end: int
    redacted_start: int
    redacted_end: int
    score: float


@dataclass
class RedactionOutput:
    redacted_text: str
    replacements: list[Replacement] = field(default_factory=list)

    @property
    def n_redacted(self) -> int:
        return len(self.replacements)


def _pseudonym(value: str, pii_type: str, key: bytes, length: int = 6) -> str:
    """Deterministic, key-dependent tag for a value."""
    digest = hmac.new(key, f"{pii_type}:{value.strip().lower()}".encode("utf-8"), sha256)
    return digest.hexdigest()[:length]


def _partial(value: str, pii_type: str, lang: str) -> str:
    tail = _PARTIAL_TAIL.get(pii_type)
    if tail is None:
        return marker_for(pii_type, lang)
    digits = re.sub(r"\D", "", value)
    if len(digits) > tail:
        return f"{marker_for(pii_type, lang)[:-1]}_****{digits[-tail:]}]"
    return marker_for(pii_type, lang)


def build_replacement(
    entity: PIIEntity,
    mode: RedactionMode | str = RedactionMode.MASK,
    lang: str = "es",
    key: bytes = b"pii-pipeline-demo-key",
) -> str:
    """Return the replacement string for one entity under ``mode``."""
    mode = RedactionMode(mode)
    marker = marker_for(entity.type, lang)
    if mode is RedactionMode.MASK:
        return marker
    if mode is RedactionMode.PARTIAL:
        return _partial(entity.text, entity.type, lang)
    return f"{marker[:-1]}:{_pseudonym(entity.text, entity.type, key)}]"


def redact(
    text: str,
    entities: Iterable[PIIEntity],
    mode: RedactionMode | str = RedactionMode.MASK,
    lang: str = "es",
    key: bytes = b"pii-pipeline-demo-key",
) -> RedactionOutput:
    """Rewrite ``text`` replacing every entity, tracking both coordinate systems.

    Substitutions are applied right to left so that an earlier replacement can
    never shift the offsets of one that has not been applied yet.
    """
    ordered = sorted(entities, key=lambda e: e.start)
    # Overlapping spans cannot both be applied; the caller is expected to have
    # resolved them, but guard here so redaction can never produce mangled text.
    safe: list[PIIEntity] = []
    for ent in ordered:
        if safe and ent.start < safe[-1].end:
            continue
        safe.append(ent)

    out = text
    pending: list[tuple[PIIEntity, str]] = [
        (ent, build_replacement(ent, mode, lang, key)) for ent in safe
    ]
    for ent, replacement in reversed(pending):
        out = out[: ent.start] + replacement + out[ent.end :]

    # Second pass left to right to record where each replacement landed.
    replacements: list[Replacement] = []
    shift = 0
    for ent, replacement in pending:
        new_start = ent.start + shift
        new_end = new_start + len(replacement)
        replacements.append(
            Replacement(
                original=ent.text,
                replacement=replacement,
                type=ent.type,
                source_start=ent.start,
                source_end=ent.end,
                redacted_start=new_start,
                redacted_end=new_end,
                score=round(ent.score, 4),
            )
        )
        shift += len(replacement) - ent.length

    return RedactionOutput(redacted_text=out, replacements=replacements)


def verify_no_leakage(
    output: RedactionOutput, min_length: int = 4
) -> list[dict[str, Any]]:
    """Assert that no redacted value survives verbatim in the output.

    This is the last line of defence: an off-by-one in span handling would
    otherwise ship a document that looks redacted but is not.
    """
    leaks: list[dict[str, Any]] = []
    for rep in output.replacements:
        value = rep.original.strip()
        if len(value) < min_length:
            continue  # too short to be a meaningful leak on its own
        if value in output.redacted_text:
            leaks.append({"type": rep.type, "value": value})
    return leaks
