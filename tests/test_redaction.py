"""Redaction must rewrite the text without corrupting offsets or leaking values."""

import pytest

from pii_pipeline.entities import PIIEntity
from pii_pipeline.redaction import RedactionMode, redact, verify_no_leakage

TEXT = "Cliente: Ana Ruiz. Email: ana@acme.es. Tarjeta: 4539 1488 0343 6467."
ENTITIES = [
    PIIEntity("Ana Ruiz", "PERSON", 9, 17, 0.95, "test"),
    PIIEntity("ana@acme.es", "EMAIL", 26, 37, 0.99, "test"),
    PIIEntity("4539 1488 0343 6467", "CREDIT_CARD", 48, 67, 0.99, "test"),
]


@pytest.mark.parametrize("mode", list(RedactionMode))
def test_no_value_survives_redaction(mode):
    out = redact(TEXT, ENTITIES, mode=mode, lang="es")
    assert verify_no_leakage(out) == []
    for ent in ENTITIES:
        assert ent.text not in out.redacted_text


def test_surrounding_text_is_preserved():
    out = redact(TEXT, ENTITIES, mode="mask", lang="es")
    assert out.redacted_text.startswith("Cliente: ")
    assert "Email: " in out.redacted_text
    assert out.redacted_text.endswith(".")


def test_replacement_offsets_point_at_the_replacement():
    out = redact(TEXT, ENTITIES, mode="mask", lang="es")
    for rep in out.replacements:
        assert (
            out.redacted_text[rep.redacted_start : rep.redacted_end] == rep.replacement
        )
        assert TEXT[rep.source_start : rep.source_end] == rep.original


def test_pseudonymisation_is_deterministic_and_type_scoped():
    a = redact(TEXT, ENTITIES, mode="pseudonymize", lang="es").redacted_text
    b = redact(TEXT, ENTITIES, mode="pseudonymize", lang="es").redacted_text
    assert a == b
    # A different key must produce different tags.
    c = redact(TEXT, ENTITIES, mode="pseudonymize", lang="es", key=b"other").redacted_text
    assert a != c


def test_overlapping_entities_do_not_mangle_text():
    overlapping = ENTITIES + [PIIEntity("Ana", "PERSON", 9, 12, 0.7, "test")]
    out = redact(TEXT, overlapping, mode="mask", lang="es")
    assert "[NOMBRE][NOMBRE]" not in out.redacted_text
    assert verify_no_leakage(out) == []


def test_partial_mode_keeps_only_a_short_tail():
    out = redact(TEXT, ENTITIES, mode="partial", lang="es")
    assert "****6467" in out.redacted_text
    assert "4539 1488 0343 6467" not in out.redacted_text
