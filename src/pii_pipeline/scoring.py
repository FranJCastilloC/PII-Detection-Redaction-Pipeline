"""Confidence fusion and calibration.

A detector's raw score answers "how sure is this component?". What the review
queue actually needs is "how often is a span with this score correct?" -- a
calibrated probability. This module does both steps:

1. **Fusion** merges the rule engine, Presidio and the model into one set of
   spans, rewarding agreement and flagging disagreement.
2. **Calibration** maps fused scores onto observed correctness with Platt
   scaling fitted on held-out data, and reports the expected calibration error
   so the improvement is measurable rather than asserted.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .detectors.base import _rank_key
from .entities import PIIEntity

#: Minimum IoU for two detections to be considered "the same span".
AGREEMENT_IOU = 0.5
#: Multiplier applied when detectors overlap but disagree on the entity type.
CONFLICT_PENALTY = 0.75


def noisy_or(scores: Sequence[float]) -> float:
    """Combine independent positive evidence: ``1 - prod(1 - s)``.

    Two detectors that independently flag the same span are jointly more
    convincing than either alone, which is the property we want here.
    """
    product = 1.0
    for s in scores:
        product *= 1.0 - min(max(s, 0.0), 1.0)
    return 1.0 - product


def _cluster(entities: list[PIIEntity]) -> list[list[PIIEntity]]:
    """Group entities into clusters of mutually overlapping spans."""
    ordered = sorted(entities, key=lambda e: (e.start, e.end))
    clusters: list[list[PIIEntity]] = []
    for ent in ordered:
        if clusters and any(ent.overlaps(other) for other in clusters[-1]):
            clusters[-1].append(ent)
        else:
            clusters.append([ent])
    return clusters


def fuse(detections: Iterable[PIIEntity]) -> list[PIIEntity]:
    """Merge detections from several detectors into one calibrated-ready set."""
    fused: list[PIIEntity] = []
    for cluster in _cluster(list(detections)):
        winner = max(cluster, key=_rank_key)
        sources = sorted({e.source for e in cluster})

        agreeing = [
            e
            for e in cluster
            if e.type == winner.type and (e is winner or e.iou(winner) >= AGREEMENT_IOU)
        ]
        conflicting = sorted({e.type for e in cluster if e.type != winner.type})

        score = noisy_or([e.score for e in agreeing]) if len(agreeing) > 1 else winner.score
        if conflicting:
            score *= CONFLICT_PENALTY

        merged = PIIEntity(
            text=winner.text,
            type=winner.type,
            start=winner.start,
            end=winner.end,
            score=score,
            source="+".join(sources),
            metadata={
                **winner.metadata,
                "detectors": sources,
                "n_agreeing": len(agreeing),
                "agreement": len(agreeing) > 1,
                "conflicting_types": conflicting,
                "raw_scores": {e.source: round(e.score, 4) for e in cluster},
            },
        )
        fused.append(merged)
    return sorted(fused, key=lambda e: e.start)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def expected_calibration_error(
    scores: Sequence[float], correct: Sequence[int], n_bins: int = 10
) -> float:
    """Weighted average gap between confidence and accuracy across bins."""
    scores_arr = np.asarray(scores, dtype=float)
    correct_arr = np.asarray(correct, dtype=float)
    if scores_arr.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (scores_arr > lo) & (scores_arr <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct_arr[mask].mean() - scores_arr[mask].mean())
    return float(ece)


def reliability_bins(
    scores: Sequence[float], correct: Sequence[int], n_bins: int = 10
) -> list[dict[str, float]]:
    """Per-bin confidence vs accuracy, for the reliability diagram."""
    scores_arr = np.asarray(scores, dtype=float)
    correct_arr = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (scores_arr > lo) & (scores_arr <= hi)
        bins.append(
            {
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "count": int(mask.sum()),
                "mean_confidence": float(scores_arr[mask].mean()) if mask.any() else 0.0,
                "accuracy": float(correct_arr[mask].mean()) if mask.any() else 0.0,
            }
        )
    return bins


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def calibration_group(entity: PIIEntity) -> str:
    """Key under which an entity's confidence is calibrated.

    Miscalibration here is not a uniform softness that one global curve can fix:
    it is concentrated in specific (detector, type) combinations. Presidio's
    spaCy NER, for instance, emits a constant 0.85 for every PERSON and LOCATION
    it finds, and on this corpus most of those are wrong (capitalised common
    nouns, Spanish verbs, HTML fragments) while a 0.85 from the rule engine is
    usually right. Averaging the two produces a curve that is non-monotonic and
    therefore uncorrectable by any single Platt or temperature fit.
    """
    detectors = entity.metadata.get("detectors") or [entity.source]
    return f"{'+'.join(sorted(detectors))}|{entity.type}"


@dataclass
class ConfidenceCalibrator:
    """Platt scaling on the logit of the fused score, fitted per group.

    Platt rather than plain temperature scaling because the fused score is not a
    softmax output: the noisy-OR fusion introduces a systematic bias as well as a
    sharpness error, and a single temperature cannot correct a bias.
    """

    slope: float = 1.0
    intercept: float = 0.0
    fitted: bool = False
    #: group key -> (slope, intercept), for groups with enough evidence.
    groups: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: Below this many samples a group falls back to the global fit, rather than
    #: fitting a confident curve on noise.
    min_group_samples: int = 40

    # -- fitting -----------------------------------------------------------
    @staticmethod
    def _fit_platt(scores: Sequence[float], correct: Sequence[int]) -> tuple[float, float] | None:
        from sklearn.linear_model import LogisticRegression

        if len(scores) < 20 or len({int(c) for c in correct}) < 2:
            return None
        X = np.array([[_logit(s)] for s in scores])
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(
            X, np.asarray(correct, dtype=int)
        )
        return float(model.coef_[0][0]), float(model.intercept_[0])

    def fit(
        self,
        scores: Sequence[float],
        correct: Sequence[int],
        groups: Sequence[str] | None = None,
    ) -> "ConfidenceCalibrator":
        global_fit = self._fit_platt(scores, correct)
        if global_fit is None:
            self.fitted = False
            return self
        self.slope, self.intercept = global_fit
        self.fitted = True

        if groups is None:
            return self
        buckets: dict[str, tuple[list[float], list[int]]] = {}
        for score, ok, key in zip(scores, correct, groups):
            s_list, c_list = buckets.setdefault(key, ([], []))
            s_list.append(score)
            c_list.append(int(ok))

        self.groups = {}
        for key, (s_list, c_list) in buckets.items():
            if len(s_list) < self.min_group_samples:
                continue
            if len(set(c_list)) < 2:
                # A group that is always right or always wrong has no curve to
                # fit; pin it near its observed rate instead.
                rate = min(max(sum(c_list) / len(c_list), 0.02), 0.98)
                self.groups[key] = (0.0, _logit(rate))
                continue
            fit = self._fit_platt(s_list, c_list)
            if fit is not None:
                self.groups[key] = fit
        return self

    # -- applying ----------------------------------------------------------
    def transform(self, score: float, group: str | None = None) -> float:
        """Calibrate ``score``, but only where there is evidence to do so.

        A group with no fitted curve is left untouched rather than pushed through
        the global fit. The global fit is dominated by the detectors that produce
        the most predictions -- here, Presidio's frequently-wrong PERSON and
        LOCATION spans -- so applying it to an unrelated, well-behaved detector
        actively destroys information. Observed before this guard: a checksum-
        verified IBAN went from 0.75 to 0.42 and was discarded, leaking the
        account number.
        """
        if not self.fitted or group not in self.groups:
            return score
        slope, intercept = self.groups[group]
        z = slope * _logit(score) + intercept
        return float(1.0 / (1.0 + math.exp(-z)))

    def apply(self, entities: Iterable[PIIEntity]) -> list[PIIEntity]:
        out: list[PIIEntity] = []
        for ent in entities:
            raw = ent.score
            group = calibration_group(ent)
            ent.metadata["raw_score"] = round(raw, 4)
            ent.metadata["calibration_group"] = group
            ent.score = self.transform(raw, group)
            ent.metadata["calibrated"] = self.fitted
            ent.metadata["group_calibrated"] = group in self.groups
            out.append(ent)
        return out

    # -- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "slope": self.slope,
                    "intercept": self.intercept,
                    "fitted": self.fitted,
                    "min_group_samples": self.min_group_samples,
                    "groups": {k: list(v) for k, v in self.groups.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "ConfidenceCalibrator":
        if not Path(path).exists():
            return cls()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            slope=payload.get("slope", 1.0),
            intercept=payload.get("intercept", 0.0),
            fitted=payload.get("fitted", False),
            groups={k: tuple(v) for k, v in payload.get("groups", {}).items()},
            min_group_samples=payload.get("min_group_samples", 40),
        )
