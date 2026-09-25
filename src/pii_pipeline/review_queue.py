"""Routing policy and persistence for the human review queue.

The point of the queue is to make the precision/recall trade-off an operational
decision instead of a modelling one. Everything the pipeline is confident about
is redacted automatically; everything it is unsure about still gets redacted,
but is flagged for a reviewer who can release a false positive. Nothing that
looks like PII is ever silently left in the document.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .entities import PIIEntity


class Decision(str, Enum):
    AUTO_REDACT = "auto_redact"
    REVIEW = "review"
    DISCARD = "discard"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"   # reviewer confirms it is PII -> stays redacted
    REJECTED = "rejected"   # reviewer says false positive -> restore original


@dataclass
class ReviewPolicy:
    """Thresholds and business rules deciding who handles each detection."""

    #: At or above this calibrated confidence, redact without asking.
    tau_auto: float = 0.90
    #: Below this, the span is too weak to act on (logged, never silently lost).
    tau_discard: float = 0.50
    #: High-risk types need near-certainty before skipping human review: the
    #: cost of a leaked card number dwarfs the cost of one extra review.
    tau_high_risk: float = 0.95
    #: Send overlapping/conflicting type assignments to a human regardless.
    review_on_conflict: bool = True
    #: Send spans only one detector saw to a human when below this confidence.
    tau_single_detector: float = 0.97

    def decide(self, entity: PIIEntity) -> tuple[Decision, list[str]]:
        """Return the routing decision for ``entity`` plus the reasons for it."""
        reasons: list[str] = []
        meta = entity.metadata

        checksum_verified = meta.get("checksum_passed") is True
        if entity.score < self.tau_discard:
            if not checksum_verified:
                return Decision.DISCARD, [f"score {entity.score:.2f} < {self.tau_discard}"]
            # A passing checksum is arithmetic proof, not an estimate. Whatever
            # the score says, this is an identifier and it does not get dropped.
            reasons.append(
                f"low score {entity.score:.2f} but {meta.get('checksum')} checksum verified"
            )

        if self.review_on_conflict and meta.get("conflicting_types"):
            reasons.append(f"type conflict with {meta['conflicting_types']}")
        if meta.get("checksum_passed") is False:
            reasons.append(f"{meta.get('checksum')} checksum failed")
        if entity.is_high_risk and entity.score < self.tau_high_risk:
            reasons.append(
                f"high-risk {entity.type} below {self.tau_high_risk} confidence"
            )
        if entity.score < self.tau_auto:
            reasons.append(f"score {entity.score:.2f} < {self.tau_auto}")
        if (
            not meta.get("agreement", False)
            and meta.get("detectors")
            and len(meta["detectors"]) == 1
            and entity.score < self.tau_single_detector
        ):
            reasons.append(f"only {meta['detectors'][0]} detected it")

        if reasons:
            return Decision.REVIEW, reasons
        return Decision.AUTO_REDACT, ["high confidence, no conflicts"]


@dataclass
class ReviewItem:
    """One queued detection awaiting a human verdict."""

    document_id: str
    entity_type: str
    text: str
    start: int
    end: int
    score: float
    reasons: list[str]
    context: str
    source: str = ""
    status: str = ReviewStatus.PENDING.value
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    resolved_at: str | None = None
    reviewer_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_context(text: str, start: int, end: int, window: int = 60) -> str:
    """Snippet around a span so a reviewer can judge without opening the file."""
    lo, hi = max(0, start - window), min(len(text), end + window)
    prefix = "..." if lo > 0 else ""
    suffix = "..." if hi < len(text) else ""
    return f"{prefix}{text[lo:start]}<<{text[start:end]}>>{text[end:hi]}{suffix}".replace(
        "\n", " "
    )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   TEXT NOT NULL,
    entity_type   TEXT NOT NULL,
    text          TEXT NOT NULL,
    start         INTEGER NOT NULL,
    end           INTEGER NOT NULL,
    score         REAL NOT NULL,
    reasons       TEXT NOT NULL,
    context       TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'pending',
    created_at    TEXT NOT NULL,
    resolved_at   TEXT,
    reviewer_note TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_review_status ON review_items(status);
CREATE INDEX IF NOT EXISTS idx_review_document ON review_items(document_id);
"""


class ReviewQueue:
    """SQLite-backed queue with an append-only JSONL audit trail beside it."""

    def __init__(self, db_path: Path | str = "data/review_queue.sqlite", jsonl_path: Path | str | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = Path(jsonl_path) if jsonl_path else self.db_path.with_suffix(".jsonl")
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def enqueue(self, items: Iterable[ReviewItem]) -> int:
        rows = [
            (
                it.document_id, it.entity_type, it.text, it.start, it.end, it.score,
                json.dumps(it.reasons, ensure_ascii=False), it.context, it.source,
                it.status, it.created_at, it.resolved_at, it.reviewer_note,
            )
            for it in items
        ]
        if not rows:
            return 0
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.executemany(
                "INSERT INTO review_items (document_id, entity_type, text, start, end,"
                " score, reasons, context, source, status, created_at, resolved_at,"
                " reviewer_note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            conn.commit()
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            for it in items:
                fh.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")
        return len(rows)

    def pending(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM review_items WHERE status = 'pending'"
                           " ORDER BY score ASC LIMIT ?", (limit,))

    def all_items(self, limit: int = 500) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM review_items ORDER BY id DESC LIMIT ?", (limit,))

    def _query(self, sql: str, params: tuple) -> list[dict[str, Any]]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["reasons"] = json.loads(item["reasons"])
            out.append(item)
        return out

    def resolve(self, item_id: int, status: ReviewStatus | str, note: str = "") -> None:
        status = ReviewStatus(status)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "UPDATE review_items SET status = ?, resolved_at = ?, reviewer_note = ?"
                " WHERE id = ?",
                (
                    status.value,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    note,
                    item_id,
                ),
            )
            conn.commit()

    def stats(self) -> dict[str, int]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) FROM review_items GROUP BY status"
            ).fetchall()
        return {status: count for status, count in rows}

    def clear(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("DELETE FROM review_items")
            conn.commit()
