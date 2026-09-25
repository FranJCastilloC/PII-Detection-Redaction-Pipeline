"""The detection systems that get compared against each other."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..detectors.base import resolve_overlaps
from ..detectors.presidio_detector import PresidioDetector
from ..detectors.regex_rules import RegexRuleDetector
from ..detectors.transformer_detector import DEFAULT_MODEL_DIR, TransformerDetector
from ..entities import PIIEntity
from ..scoring import ConfidenceCalibrator, fuse

System = Callable[[str, str], list[PIIEntity]]


def build_systems(
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    calibrator: ConfidenceCalibrator | None = None,
    max_length: int = 192,
) -> dict[str, System]:
    """Return the named systems to evaluate, skipping the model if absent.

    The comparison is deliberately laddered: each rung adds exactly one
    component, so any change in the numbers can be attributed to it.
    """
    rules = RegexRuleDetector()
    presidio = PresidioDetector()
    model = TransformerDetector(model_dir, max_length=max_length)
    calibrator = calibrator or ConfidenceCalibrator()

    systems: dict[str, System] = {
        "rules": lambda text, lang: rules.detect(text, lang),
        "presidio": lambda text, lang: presidio.detect(text, lang),
        "baseline_rules_presidio": lambda text, lang: resolve_overlaps(
            fuse(rules.detect(text, lang) + presidio.detect(text, lang))
        ),
    }

    if model.is_available:
        systems["finetuned_model"] = lambda text, lang: model.detect(text, lang)

        def ensemble(text: str, lang: str) -> list[PIIEntity]:
            # Deliberately identical to what PIIPipeline runs in production,
            # document-consistency steps included: evaluating a stripped-down
            # variant would report numbers for a system nobody ships.
            from ..pipeline import _unify_repeated_values, sweep_repeated_occurrences

            merged = calibrator.apply(
                resolve_overlaps(
                    fuse(
                        rules.detect(text, lang)
                        + presidio.detect(text, lang)
                        + model.detect(text, lang)
                    )
                )
            )
            return sweep_repeated_occurrences(text, _unify_repeated_values(merged))

        systems["ensemble"] = ensemble
    return systems


def to_spans(entities: list[PIIEntity], min_score: float = 0.0) -> list[dict]:
    """Project entities onto the plain span dicts the metrics module expects."""
    from ..scoring import calibration_group

    return [
        {
            "start": e.start,
            "end": e.end,
            "type": e.type,
            "score": e.score,
            # Carried through so the calibrator can be fitted per (detector, type).
            "group": calibration_group(e),
        }
        for e in entities
        if e.score >= min_score
    ]
