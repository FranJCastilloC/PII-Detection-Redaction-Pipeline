"""Rule detection, span algebra and review-queue routing."""

from pii_pipeline.detectors.base import resolve_overlaps
from pii_pipeline.detectors.regex_rules import RegexRuleDetector
from pii_pipeline.entities import PIIEntity
from pii_pipeline.review_queue import Decision, ReviewPolicy
from pii_pipeline.scoring import fuse, noisy_or

detector = RegexRuleDetector()


def _types(text, lang="es"):
    return {e.type: e.text for e in detector.detect(text, lang)}


def test_detects_core_spanish_entities():
    found = _types(
        "DNI: 12345678Z, tel 612 345 678, email a.ruiz@acme.es, IP 10.0.0.4"
    )
    assert found["GOV_ID"] == "12345678Z"
    assert found["EMAIL"] == "a.ruiz@acme.es"
    assert found["PHONE"] == "612 345 678"
    assert found["IP_ADDRESS"] == "10.0.0.4"


def test_valid_checksum_outranks_a_failing_one_on_overlap():
    # A digit run inside an IBAN also matches the card pattern and fails Luhn;
    # the verified IBAN must win, or the pipeline leaks the account number.
    found = _types("Tarjeta y cuenta IBAN: ES91 2100 0418 4502 0005 1332")
    assert found.get("BANK_ACCOUNT") == "ES91 2100 0418 4502 0005 1332"


def test_entities_never_span_a_line_break():
    text = "Cliente: Maria Rodriguez\nDNI: 12345678Z"
    for ent in detector.detect(text, "es"):
        if ent.type != "ADDRESS":
            assert "\n" not in ent.text


def test_birth_date_context_does_not_leak_to_the_next_line():
    text = "Fecha de nacimiento: 14/03/1985\nFecha de emision: 12/09/2024"
    found = {e.text: e.type for e in detector.detect(text, "es")}
    assert found["14/03/1985"] == "DATE_OF_BIRTH"
    assert found["12/09/2024"] == "DATE_TIME"


def test_noisy_or_rewards_agreement():
    assert noisy_or([0.8]) == 0.8
    assert noisy_or([0.8, 0.8]) > 0.9


def test_fusion_flags_type_conflicts_and_penalises_them():
    a = PIIEntity("123456", "GOV_ID", 0, 6, 0.9, "rules")
    b = PIIEntity("123456", "PHONE", 0, 6, 0.8, "model")
    fused = fuse([a, b])
    assert len(fused) == 1
    assert fused[0].metadata["conflicting_types"]
    assert fused[0].score < 0.9


def test_resolve_overlaps_keeps_one_span_per_cluster():
    kept = resolve_overlaps([
        PIIEntity("abc", "PERSON", 0, 3, 0.6, "a"),
        PIIEntity("abcdef", "PERSON", 0, 6, 0.9, "b"),
    ])
    assert len(kept) == 1
    assert kept[0].length == 6


def test_policy_routes_by_confidence_and_risk():
    policy = ReviewPolicy()
    strong = PIIEntity("a@b.es", "EMAIL", 0, 6, 0.99, "x",
                       metadata={"detectors": ["rules", "model"], "agreement": True})
    assert policy.decide(strong)[0] is Decision.AUTO_REDACT

    weak = PIIEntity("Juan", "PERSON", 0, 4, 0.60, "x",
                     metadata={"detectors": ["rules"], "agreement": False})
    assert policy.decide(weak)[0] is Decision.REVIEW

    tiny = PIIEntity("x", "PERSON", 0, 1, 0.10, "x")
    assert policy.decide(tiny)[0] is Decision.DISCARD


def test_high_risk_needs_near_certainty_to_skip_review():
    policy = ReviewPolicy()
    card = PIIEntity("4539148803436467", "CREDIT_CARD", 0, 16, 0.92, "x",
                     metadata={"detectors": ["rules", "model"], "agreement": True})
    decision, reasons = policy.decide(card)
    assert decision is Decision.REVIEW
    assert any("high-risk" in r for r in reasons)


def test_merge_adjacent_spans_normalises_annotation_granularity():
    from pii_pipeline.evaluation.metrics import merge_adjacent_spans

    text = "Mayza Balloi vive en Conygre Grove, 163"
    # The corpus annotates given name and surname as two separate PERSON spans.
    gold = [
        {"start": 0, "end": 5, "type": "PERSON", "score": 0.9},
        {"start": 6, "end": 12, "type": "PERSON", "score": 0.7},
        {"start": 21, "end": 34, "type": "ADDRESS"},
        {"start": 36, "end": 39, "type": "ADDRESS"},
    ]
    merged = merge_adjacent_spans(gold, text)
    assert len(merged) == 2
    assert text[merged[0]["start"] : merged[0]["end"]] == "Mayza Balloi"
    assert merged[0]["score"] == 0.7  # weakest part wins
    assert text[merged[1]["start"] : merged[1]["end"]] == "Conygre Grove, 163"


def test_merge_does_not_join_across_unrelated_text():
    from pii_pipeline.evaluation.metrics import merge_adjacent_spans

    text = "Ana trabaja con Juan"
    spans = [
        {"start": 0, "end": 3, "type": "PERSON"},
        {"start": 16, "end": 20, "type": "PERSON"},
    ]
    assert len(merge_adjacent_spans(spans, text)) == 2


def test_repeated_values_get_one_consistent_decision():
    from pii_pipeline.pipeline import _unify_repeated_values

    # Same string, two mentions, scores straddling the discard threshold.
    ents = [
        PIIEntity("ORD-300599", "GOV_ID", 98, 108, 0.51, "rules"),
        PIIEntity("ORD-300599", "GOV_ID", 169, 179, 0.48, "rules"),
    ]
    unified = _unify_repeated_values(ents)
    assert {round(e.score, 2) for e in unified} == {0.51}

    policy = ReviewPolicy()
    assert len({policy.decide(e)[0] for e in unified}) == 1


def test_sweep_catches_undetected_repeats_of_a_redacted_value():
    from pii_pipeline.pipeline import sweep_repeated_occurrences

    text = "Ticket TCK-33981 abierto. Referencia interna: TCK-33981."
    detected = [PIIEntity("TCK-33981", "GOV_ID", 46, 55, 0.6, "rules")]
    swept = sweep_repeated_occurrences(text, detected)
    assert len(swept) == 2
    assert all(text[e.start : e.end] == "TCK-33981" for e in swept)


def test_sweep_does_not_fire_inside_a_longer_word():
    from pii_pipeline.pipeline import sweep_repeated_occurrences

    text = "El usuario ana escribio anagrama y anatomia."
    detected = [PIIEntity("ana", "USERNAME", 11, 14, 0.9, "rules")]
    # "ana" is under the minimum length and also embedded in other words.
    assert len(sweep_repeated_occurrences(text, detected)) == 1


def test_calibrator_leaves_ungrouped_scores_untouched():
    from pii_pipeline.scoring import ConfidenceCalibrator

    cal = ConfidenceCalibrator(slope=0.88, intercept=-1.36, fitted=True,
                               groups={"rules|EMAIL": (1.0, 0.0)})
    # No curve fitted for this group: the score must survive unchanged rather
    # than absorb a bias estimated from a different detector population.
    assert cal.transform(0.75, "rules|BANK_ACCOUNT") == 0.75


def test_checksum_verified_entity_is_never_discarded():
    policy = ReviewPolicy()
    iban = PIIEntity("ES3478720097447379727611", "BANK_ACCOUNT", 0, 24, 0.42, "rules",
                     metadata={"checksum": "iban", "checksum_passed": True})
    decision, reasons = policy.decide(iban)
    assert decision is Decision.REVIEW
    assert any("checksum verified" in r for r in reasons)


def test_username_rule_needs_a_separator_and_skips_the_qualifier():
    found = {e.type: e.text for e in detector.detect("Usuario deseado: rferrandez46", "es")}
    assert found["USERNAME"] == "rferrandez46"
    # No separator at all: the label's own words must not be captured.
    assert not [e for e in detector.detect("El usuario principal es clave", "es")
                if e.type == "USERNAME"]
