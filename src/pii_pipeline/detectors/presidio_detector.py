"""Microsoft Presidio wrapper, part of the baseline alongside the regex rules.

Presidio contributes what handwritten patterns cannot: spaCy statistical NER for
person names and locations, in both languages. Its own recogniser labels are
remapped onto this project's 12-class taxonomy so the comparison against the
fine-tuned model is apples to apples.
"""

from __future__ import annotations

import logging

from ..entities import PIIEntity, PIIType
from .base import normalize_entities, resolve_overlaps

logger = logging.getLogger(__name__)

#: spaCy pipelines backing each language, best first. The ``md`` models are
#: noticeably better at PERSON/LOCATION; the ``sm`` ones are ~12 MB instead of
#: ~50 MB and exist so the hosted demo fits inside a 1 GB container. Whichever
#: is actually installed wins, so the same code runs locally and in the cloud.
SPACY_MODELS: dict[str, tuple[str, ...]] = {
    "en": ("en_core_web_md", "en_core_web_sm"),
    "es": ("es_core_news_md", "es_core_news_sm"),
}


def resolve_spacy_model(lang: str) -> str:
    """Return the best spaCy model for ``lang`` that is actually installed.

    Presidio will otherwise try to *download* a missing model at request time,
    which on a hosted container blocks the request forever instead of failing.
    """
    import importlib.util

    candidates = SPACY_MODELS.get(lang, ())
    for name in candidates:
        if importlib.util.find_spec(name) is not None:
            return name
    raise LookupError(
        f"no spaCy model installed for {lang!r}; expected one of {candidates}. "
        f"Install with: python -m spacy download {candidates[0]}"
    )

#: Presidio recogniser label -> project taxonomy. ``None`` means "drop".
PRESIDIO_LABEL_MAP: dict[str, str | None] = {
    "PERSON": PIIType.PERSON.value,
    "EMAIL_ADDRESS": PIIType.EMAIL.value,
    "PHONE_NUMBER": PIIType.PHONE.value,
    "LOCATION": PIIType.ADDRESS.value,
    "ADDRESS": PIIType.ADDRESS.value,
    "CREDIT_CARD": PIIType.CREDIT_CARD.value,
    "CRYPTO": PIIType.BANK_ACCOUNT.value,
    "IBAN_CODE": PIIType.BANK_ACCOUNT.value,
    "US_BANK_NUMBER": PIIType.BANK_ACCOUNT.value,
    "US_SSN": PIIType.GOV_ID.value,
    "US_ITIN": PIIType.GOV_ID.value,
    "US_PASSPORT": PIIType.GOV_ID.value,
    "US_DRIVER_LICENSE": PIIType.GOV_ID.value,
    "UK_NHS": PIIType.GOV_ID.value,
    "UK_NINO": PIIType.GOV_ID.value,
    "ES_NIF": PIIType.GOV_ID.value,
    "ES_NIE": PIIType.GOV_ID.value,
    "IT_FISCAL_CODE": PIIType.GOV_ID.value,
    "IT_DRIVER_LICENSE": PIIType.GOV_ID.value,
    "IT_PASSPORT": PIIType.GOV_ID.value,
    "IT_IDENTITY_CARD": PIIType.GOV_ID.value,
    "MEDICAL_LICENSE": PIIType.GOV_ID.value,
    "AU_ABN": PIIType.GOV_ID.value,
    "AU_ACN": PIIType.GOV_ID.value,
    "AU_TFN": PIIType.GOV_ID.value,
    "AU_MEDICARE": PIIType.GOV_ID.value,
    "IN_PAN": PIIType.GOV_ID.value,
    "IN_AADHAAR": PIIType.GOV_ID.value,
    "IP_ADDRESS": PIIType.IP_ADDRESS.value,
    "DATE_TIME": PIIType.DATE_TIME.value,
    "URL": None,
    "NRP": None,  # nationality / religion / political group: out of scope
}


class PresidioDetector:
    """Adapter exposing Presidio through the project's ``Detector`` protocol."""

    name = "presidio"

    def __init__(self, languages: tuple[str, ...] = ("es", "en"), score_floor: float = 0.3):
        self.languages = languages
        self.score_floor = score_floor
        self._analyzer = None

    # -- lazy init ---------------------------------------------------------
    @property
    def analyzer(self):
        """Build the analyzer on first use (loading two spaCy models is slow)."""
        if self._analyzer is None:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            configuration = {
                "nlp_engine_name": "spacy",
                "models": [
                    {"lang_code": lang, "model_name": resolve_spacy_model(lang)}
                    for lang in self.languages
                ],
            }
            nlp_engine = NlpEngineProvider(nlp_configuration=configuration).create_engine()
            self._analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine, supported_languages=list(self.languages)
            )
        return self._analyzer

    @property
    def is_available(self) -> bool:
        """True when every language has a usable spaCy model installed."""
        try:
            for lang in self.languages:
                resolve_spacy_model(lang)
        except LookupError:
            return False
        return True

    # -- main --------------------------------------------------------------
    def detect(self, text: str, lang: str = "es") -> list[PIIEntity]:
        if lang not in self.languages:
            lang = self.languages[0]
        try:
            results = self.analyzer.analyze(text=text, language=lang)
        except Exception:  # pragma: no cover - depends on optional models
            logger.exception("presidio analysis failed; returning no entities")
            return []

        entities: list[PIIEntity] = []
        for res in results:
            mapped = PRESIDIO_LABEL_MAP.get(res.entity_type, "__unmapped__")
            if mapped is None:
                continue
            if mapped == "__unmapped__":
                logger.debug("unmapped presidio label %s", res.entity_type)
                continue
            if res.score < self.score_floor:
                continue
            entities.append(
                PIIEntity(
                    text=text[res.start : res.end],
                    type=mapped,
                    start=res.start,
                    end=res.end,
                    score=float(res.score),
                    source=self.name,
                    metadata={"presidio_label": res.entity_type},
                )
            )
        return resolve_overlaps(normalize_entities(text, entities))
