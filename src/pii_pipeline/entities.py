"""PII taxonomy, entity container and label mappings.

The project deliberately collapses the ~25 heterogeneous labels shipped by the
ai4privacy corpus into 12 *actionable* PII classes: the ones a business-document
redaction workflow is legally and operationally required to remove. Labels that
carry no direct re-identification risk on their own (job title, gender, currency,
...) are mapped to ``O``; :mod:`pii_pipeline.data.build_dataset` reports every
label it dropped so the decision stays auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class PIIType(str, Enum):
    """The 12 actionable PII classes handled end to end by the pipeline."""

    PERSON = "PERSON"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    ADDRESS = "ADDRESS"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    CREDIT_CARD = "CREDIT_CARD"
    BANK_ACCOUNT = "BANK_ACCOUNT"
    GOV_ID = "GOV_ID"
    USERNAME = "USERNAME"
    CREDENTIAL = "CREDENTIAL"
    IP_ADDRESS = "IP_ADDRESS"
    DATE_TIME = "DATE_TIME"


PII_TYPES: list[str] = [t.value for t in PIIType]

# ---------------------------------------------------------------------------
# Risk tiers
# ---------------------------------------------------------------------------
# High-risk entities are the ones whose leakage is immediately monetisable by an
# attacker (payment fraud, identity theft, account takeover). They get a stricter
# review threshold and win overlap conflicts against lower-risk types.
HIGH_RISK_TYPES: frozenset[str] = frozenset(
    {
        PIIType.CREDIT_CARD.value,
        PIIType.BANK_ACCOUNT.value,
        PIIType.GOV_ID.value,
        PIIType.CREDENTIAL.value,
    }
)

# Used to break ties when two detectors claim overlapping character spans.
# Higher wins. Structured identifiers beat free-text guesses because they are
# checksum-verifiable, and a false positive there is far cheaper than a leak.
TYPE_PRIORITY: dict[str, int] = {
    PIIType.CREDENTIAL.value: 100,
    PIIType.CREDIT_CARD.value: 95,
    PIIType.BANK_ACCOUNT.value: 90,
    PIIType.GOV_ID.value: 85,
    PIIType.EMAIL.value: 80,
    PIIType.IP_ADDRESS.value: 75,
    PIIType.PHONE.value: 70,
    PIIType.DATE_OF_BIRTH.value: 60,
    PIIType.PERSON.value: 55,
    PIIType.ADDRESS.value: 50,
    PIIType.USERNAME.value: 45,
    PIIType.DATE_TIME.value: 20,
}

# ---------------------------------------------------------------------------
# Redaction markers
# ---------------------------------------------------------------------------
REDACTION_MARKERS: dict[str, dict[str, str]] = {
    "es": {
        PIIType.PERSON.value: "[NOMBRE]",
        PIIType.EMAIL.value: "[EMAIL]",
        PIIType.PHONE.value: "[TELEFONO]",
        PIIType.ADDRESS.value: "[DIRECCION]",
        PIIType.DATE_OF_BIRTH.value: "[FECHA_NACIMIENTO]",
        PIIType.CREDIT_CARD.value: "[TARJETA]",
        PIIType.BANK_ACCOUNT.value: "[CUENTA_BANCARIA]",
        PIIType.GOV_ID.value: "[IDENTIFICACION]",
        PIIType.USERNAME.value: "[USUARIO]",
        PIIType.CREDENTIAL.value: "[CREDENCIAL]",
        PIIType.IP_ADDRESS.value: "[IP]",
        PIIType.DATE_TIME.value: "[FECHA]",
    },
    "en": {
        PIIType.PERSON.value: "[NAME]",
        PIIType.EMAIL.value: "[EMAIL]",
        PIIType.PHONE.value: "[PHONE]",
        PIIType.ADDRESS.value: "[ADDRESS]",
        PIIType.DATE_OF_BIRTH.value: "[DATE_OF_BIRTH]",
        PIIType.CREDIT_CARD.value: "[CREDIT_CARD]",
        PIIType.BANK_ACCOUNT.value: "[BANK_ACCOUNT]",
        PIIType.GOV_ID.value: "[GOV_ID]",
        PIIType.USERNAME.value: "[USERNAME]",
        PIIType.CREDENTIAL.value: "[CREDENTIAL]",
        PIIType.IP_ADDRESS.value: "[IP]",
        PIIType.DATE_TIME.value: "[DATE]",
    },
}

# Colour per type, shared by the Streamlit highlighter and the matplotlib report
# so a reader recognises a class by colour across the whole project.
TYPE_COLORS: dict[str, str] = {
    PIIType.PERSON.value: "#4C78A8",
    PIIType.EMAIL.value: "#72B7B2",
    PIIType.PHONE.value: "#54A24B",
    PIIType.ADDRESS.value: "#EECA3B",
    PIIType.DATE_OF_BIRTH.value: "#B279A2",
    PIIType.CREDIT_CARD.value: "#E45756",
    PIIType.BANK_ACCOUNT.value: "#D67195",
    PIIType.GOV_ID.value: "#F58518",
    PIIType.USERNAME.value: "#9D755D",
    PIIType.CREDENTIAL.value: "#A5200B",
    PIIType.IP_ADDRESS.value: "#79706E",
    PIIType.DATE_TIME.value: "#BAB0AC",
}


def marker_for(pii_type: str, lang: str = "es") -> str:
    """Return the redaction marker for ``pii_type`` in ``lang`` (fallback: ES)."""
    table = REDACTION_MARKERS.get(lang, REDACTION_MARKERS["es"])
    return table.get(pii_type, f"[{pii_type}]")


# ---------------------------------------------------------------------------
# ai4privacy -> project taxonomy
# ---------------------------------------------------------------------------
AI4PRIVACY_LABEL_MAP: dict[str, str | None] = {
    # --- people: the corpus splits names into numbered given/last name slots ---
    "GIVENNAME1": PIIType.PERSON.value,
    "GIVENNAME2": PIIType.PERSON.value,
    "LASTNAME1": PIIType.PERSON.value,
    "LASTNAME2": PIIType.PERSON.value,
    "LASTNAME3": PIIType.PERSON.value,
    # --- contact ---
    "EMAIL": PIIType.EMAIL.value,
    "TEL": PIIType.PHONE.value,
    # --- postal address components ---
    "STREET": PIIType.ADDRESS.value,
    "BUILDING": PIIType.ADDRESS.value,
    "SECADDRESS": PIIType.ADDRESS.value,
    "CITY": PIIType.ADDRESS.value,
    "STATE": PIIType.ADDRESS.value,
    "POSTCODE": PIIType.ADDRESS.value,
    "GEOCOORD": PIIType.ADDRESS.value,
    # --- dates ---
    "BOD": PIIType.DATE_OF_BIRTH.value,
    "DATE": PIIType.DATE_TIME.value,
    "TIME": PIIType.DATE_TIME.value,
    # --- government / identity documents ---
    "SOCIALNUMBER": PIIType.GOV_ID.value,
    "IDCARD": PIIType.GOV_ID.value,
    "PASSPORT": PIIType.GOV_ID.value,
    "DRIVERLICENSE": PIIType.GOV_ID.value,
    # --- online identity ---
    "USERNAME": PIIType.USERNAME.value,
    "PASS": PIIType.CREDENTIAL.value,
    "IP": PIIType.IP_ADDRESS.value,
    # --- explicitly out of scope (see INTENTIONALLY_DROPPED) ---
    "TITLE": None,
    "SEX": None,
    "COUNTRY": None,
    "CARDISSUER": None,
}

#: Labels we intentionally drop, with the reason. They are demographic or overly
#: coarse attributes: redacting them would damage the business meaning of the
#: document without meaningfully reducing re-identification risk.
INTENTIONALLY_DROPPED: dict[str, str] = {
    "TITLE": "honorific (Mr./Sra.); not identifying on its own",
    "SEX": "demographic attribute, not a direct identifier",
    "COUNTRY": "too coarse to identify a person",
    "CARDISSUER": "brand of the card issuer, not the card number",
}

#: Classes the ai4privacy corpus simply does not contain: it annotates a card
#: *issuer* but never a card number, and no bank account numbers at all. The
#: fine-tuned model therefore cannot learn them, and the hybrid pipeline hands
#: both to the checksum-backed rule engine instead. This is stated up front
#: rather than papered over with a fabricated training signal.
TYPES_WITHOUT_CORPUS_SUPPORT: frozenset[str] = frozenset(
    {PIIType.CREDIT_CARD.value, PIIType.BANK_ACCOUNT.value}
)

#: The 10 classes the transformer is actually trained on.
MODEL_TYPES: list[str] = [t for t in PII_TYPES if t not in TYPES_WITHOUT_CORPUS_SUPPORT]


def map_source_label(label: str) -> str | None:
    """Map a raw ai4privacy label to our taxonomy, or ``None`` to drop it."""
    return AI4PRIVACY_LABEL_MAP.get(label.strip().upper())


# ---------------------------------------------------------------------------
# BIO label space
# ---------------------------------------------------------------------------
def build_bio_labels(types: list[str] | None = None) -> list[str]:
    """Return the ordered BIO label space: ``O`` plus B-/I- for each PII type."""
    labels = ["O"]
    for t in types if types is not None else PII_TYPES:
        labels.extend([f"B-{t}", f"I-{t}"])
    return labels


#: Full taxonomy label space (all 12 classes), used by the pipeline as a whole.
BIO_LABELS: list[str] = build_bio_labels()

#: Label space the transformer is trained on: the 10 classes present in the
#: corpus. Allocating output neurons for classes with zero training examples
#: would only add dead capacity and noise to the comparison.
MODEL_BIO_LABELS: list[str] = build_bio_labels(MODEL_TYPES)
LABEL2ID: dict[str, int] = {lab: i for i, lab in enumerate(MODEL_BIO_LABELS)}
ID2LABEL: dict[int, str] = {i: lab for lab, i in LABEL2ID.items()}


# ---------------------------------------------------------------------------
# Entity container
# ---------------------------------------------------------------------------
@dataclass
class PIIEntity:
    """A single detected PII span, in character offsets of the source text."""

    text: str
    type: str
    start: int
    end: int
    score: float = 1.0
    source: str = "unknown"
    #: Free-form provenance: matched rule name, checksum result, agreeing
    #: detectors, calibration inputs. Everything the audit log needs.
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"invalid span [{self.start}, {self.end}) for {self.text!r}")
        self.score = float(min(1.0, max(0.0, self.score)))

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def is_high_risk(self) -> bool:
        return self.type in HIGH_RISK_TYPES

    @property
    def priority(self) -> int:
        return TYPE_PRIORITY.get(self.type, 0)

    def overlaps(self, other: "PIIEntity") -> bool:
        return self.start < other.end and other.start < self.end

    def iou(self, other: "PIIEntity") -> float:
        """Intersection-over-union of the two character spans."""
        inter = max(0, min(self.end, other.end) - max(self.start, other.start))
        if inter == 0:
            return 0.0
        union = max(self.end, other.end) - min(self.start, other.start)
        return inter / union

    def marker(self, lang: str = "es") -> str:
        return marker_for(self.type, lang)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PIIEntity":
        known = {"text", "type", "start", "end", "score", "source", "metadata"}
        return cls(**{k: v for k, v in payload.items() if k in known})
