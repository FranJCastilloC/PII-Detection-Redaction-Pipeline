"""Span-level evaluation, plus the two metrics a privacy team actually asks for.

Standard precision/recall/F1 answers "is the model good?". It does not answer
the two questions that decide whether a redaction system can be deployed:

``leakage_rate``
    What fraction of real PII characters survived into the released document?
    This is the metric with legal consequences; recall on exact span boundaries
    is a poor proxy for it, because a span that is 90% covered still leaks.

``over_redaction_rate``
    What fraction of non-PII characters were destroyed? This is what makes the
    output unusable for the business even when privacy is perfect.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

Span = dict[str, Any]


#: Characters allowed to sit between two spans that get merged into one.
_GAP_CHARS = set(" \t,.-/")


def merge_adjacent_spans(
    spans: Sequence[Span], text: str = "", max_gap: int = 2
) -> list[Span]:
    """Join neighbouring spans of the same type into a single span.

    The taxonomy collapses labels (``GIVENNAME1`` and ``LASTNAME1`` both become
    ``PERSON``; street, city and postcode all become ``ADDRESS``), but the corpus
    still annotates the pieces separately. A detector that returns the whole name
    as one span is *right*, yet scores zero against fragmented gold because the
    IoU of "Mayza Balloi" against "Mayza" is only 0.42.

    Normalising both sides the same way removes that artefact and matches what
    redaction actually needs: one span per identifier, not one per word.
    """
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s["start"], s["end"]))
    merged: list[Span] = [dict(ordered[0])]
    for span in ordered[1:]:
        prev = merged[-1]
        gap_text = text[prev["end"] : span["start"]] if text else ""
        gap_ok = (
            span["start"] - prev["end"] <= max_gap
            and (not text or set(gap_text) <= _GAP_CHARS)
        )
        if span["type"] == prev["type"] and gap_ok and span["start"] >= prev["start"]:
            prev["end"] = max(prev["end"], span["end"])
            if "score" in prev and "score" in span:
                # The merged span is only as trustworthy as its weakest part.
                prev["score"] = min(prev["score"], span["score"])
        else:
            merged.append(dict(span))
    return merged


def _iou(a: Span, b: Span) -> float:
    inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    if inter == 0:
        return 0.0
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return inter / union


def match_spans(
    gold: Sequence[Span],
    pred: Sequence[Span],
    mode: str = "exact",
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedily align predictions to gold spans of the same type.

    Returns ``(matched_pairs, unmatched_gold_idx, unmatched_pred_idx)``. In
    ``partial`` mode a prediction counts when it overlaps the gold span by at
    least ``iou_threshold`` -- reported alongside exact match because boundary
    disagreements ("Calle Mayor 7" vs "Calle Mayor 7 bajo") are a very different
    failure from missing the entity altogether.
    """
    used_pred: set[int] = set()
    matched: list[tuple[int, int]] = []
    unmatched_gold: list[int] = []

    for gi, g in enumerate(gold):
        best_pi, best_score = None, 0.0
        for pi, p in enumerate(pred):
            if pi in used_pred or p["type"] != g["type"]:
                continue
            if mode == "exact":
                if p["start"] == g["start"] and p["end"] == g["end"]:
                    best_pi, best_score = pi, 1.0
                    break
            else:
                score = _iou(g, p)
                if score >= iou_threshold and score > best_score:
                    best_pi, best_score = pi, score
        if best_pi is None:
            unmatched_gold.append(gi)
        else:
            used_pred.add(best_pi)
            matched.append((gi, best_pi))

    unmatched_pred = [pi for pi in range(len(pred)) if pi not in used_pred]
    return matched, unmatched_gold, unmatched_pred


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
    }


def _char_set(spans: Iterable[Span]) -> set[int]:
    covered: set[int] = set()
    for s in spans:
        covered.update(range(s["start"], s["end"]))
    return covered


def evaluate(
    documents: Sequence[dict[str, Any]],
    mode: str = "exact",
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Score a list of ``{"text", "gold", "pred"}`` documents.

    ``gold`` and ``pred`` are lists of ``{"start", "end", "type"}`` spans.
    """
    per_type: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    leaked_chars = covered_gold_chars = total_gold_chars = 0
    over_chars = total_non_pii_chars = 0
    leaked_entities = total_entities = 0

    for doc in documents:
        text = doc.get("text", "")
        # Both sides are normalised identically, so no system gains or loses from
        # the corpus's per-word annotation granularity.
        gold = merge_adjacent_spans(doc.get("gold", []), text)
        pred = merge_adjacent_spans(doc.get("pred", []), text)
        matched, miss_gold, extra_pred = match_spans(gold, pred, mode, iou_threshold)

        for gi, _ in matched:
            per_type[gold[gi]["type"]]["tp"] += 1
        for gi in miss_gold:
            per_type[gold[gi]["type"]]["fn"] += 1
        for pi in extra_pred:
            per_type[pred[pi]["type"]]["fp"] += 1

        # Character-level privacy accounting, independent of span alignment.
        text_len = len(text) or (
            max([s["end"] for s in gold + pred], default=0)
        )
        gold_chars = _char_set(gold)
        pred_chars = _char_set(pred)
        total_gold_chars += len(gold_chars)
        covered_gold_chars += len(gold_chars & pred_chars)
        leaked_chars += len(gold_chars - pred_chars)
        total_non_pii_chars += max(0, text_len - len(gold_chars))
        over_chars += len(pred_chars - gold_chars)

        for span in gold:
            total_entities += 1
            span_chars = set(range(span["start"], span["end"]))
            if span_chars - pred_chars:
                leaked_entities += 1

    types = sorted(per_type)
    by_type = {t: _prf(**per_type[t]) for t in types}
    tp = sum(per_type[t]["tp"] for t in types)
    fp = sum(per_type[t]["fp"] for t in types)
    fn = sum(per_type[t]["fn"] for t in types)

    macro = {
        key: round(
            sum(by_type[t][key] for t in types) / len(types), 4
        )
        if types
        else 0.0
        for key in ("precision", "recall", "f1")
    }

    return {
        "match_mode": mode,
        "iou_threshold": iou_threshold if mode == "partial" else None,
        "micro": _prf(tp, fp, fn),
        "macro": macro,
        "by_type": by_type,
        "privacy": {
            # Share of PII characters that survived into the released document.
            "char_leakage_rate": round(leaked_chars / total_gold_chars, 4)
            if total_gold_chars
            else 0.0,
            # Share of PII entities that were not fully covered by a redaction.
            "entity_leakage_rate": round(leaked_entities / total_entities, 4)
            if total_entities
            else 0.0,
            "char_coverage": round(covered_gold_chars / total_gold_chars, 4)
            if total_gold_chars
            else 0.0,
            # Share of legitimate, non-PII characters destroyed by redaction.
            "over_redaction_rate": round(over_chars / total_non_pii_chars, 6)
            if total_non_pii_chars
            else 0.0,
            "n_entities": total_entities,
            "n_leaked_entities": leaked_entities,
        },
    }
