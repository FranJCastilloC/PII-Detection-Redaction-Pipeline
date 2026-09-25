"""Rule-based PII baseline: regular expressions + checksums + context windows.

This is the honest baseline the fine-tuned model has to beat. It is genuinely
strong on structured identifiers (a validated IBAN is a certainty, not a guess)
and genuinely weak on free-text entities such as person names and addresses,
which is exactly the trade-off the evaluation is meant to expose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from ..entities import PIIEntity, PIIType
from .base import normalize_entities, resolve_overlaps
from .validators import VALIDATORS

#: How far around a match we look for a disambiguating keyword.
CONTEXT_WINDOW = 45


@dataclass(frozen=True)
class Rule:
    """A single detection rule."""

    name: str
    pattern: re.Pattern[str]
    pii_type: str
    base_score: float
    #: Name of a checksum in :data:`~.validators.VALIDATORS`, if any.
    validator: str | None = None
    #: Score used when the regex matches but the checksum fails. ``None`` drops
    #: the match entirely.
    on_invalid_score: float | None = 0.55
    #: Lowercase keywords that, when found nearby, raise confidence.
    context: tuple[str, ...] = ()
    #: When True the match is discarded unless a context keyword is present.
    require_context: bool = False
    #: Capture group holding the actual entity (1 when the regex needs a
    #: non-captured prefix such as a label or honorific).
    group: int = 0
    langs: tuple[str, ...] = ("es", "en")
    #: Optional extra predicate on the matched string.
    guard: Callable[[str], bool] | None = field(default=None, compare=False)


def _re(pattern: str, flags: int = re.IGNORECASE) -> re.Pattern[str]:
    return re.compile(pattern, flags)


# --- helper fragments -------------------------------------------------------
_MONTHS = (
    r"(?:ene|feb|mar|abr|may|jun|jul|ago|sep|sept|oct|nov|dic"
    r"|jan|apr|aug|dec"
    r"|enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre"
    r"|noviembre|diciembre"
    r"|january|february|march|april|june|july|august|september|october"
    r"|november|december)"
)
_DOB_CONTEXT = (
    "nacimiento", "nacid", "f. nac", "fec. nac", "fnac", "born", "birth",
    "date of birth", "dob", "cumple",
)

RULES: list[Rule] = [
    # ---------------------------------------------------------------- email
    Rule(
        name="email",
        pattern=_re(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,63}\b"),
        pii_type=PIIType.EMAIL.value,
        base_score=0.97,
    ),
    # ------------------------------------------------------------ credentials
    Rule(
        name="password_labeled",
        pattern=_re(
            r"(?:contrase[nñ]a|clave|password|passwd|pwd|api[_ -]?key|token|secret)"
            r"\s*[:=]\s*(\S{6,64})"
        ),
        pii_type=PIIType.CREDENTIAL.value,
        base_score=0.93,
        group=1,
    ),
    Rule(
        name="api_key_like",
        pattern=_re(r"\b(?:sk|pk|ghp|xox[baprs])[-_][A-Za-z0-9]{16,}\b", re.NOFLAG),
        pii_type=PIIType.CREDENTIAL.value,
        base_score=0.9,
    ),
    # ------------------------------------------------------------- financial
    Rule(
        name="iban",
        pattern=_re(r"\b[A-Z]{2}\d{2}[\s-]?(?:[A-Z0-9]{4}[\s-]?){2,7}[A-Z0-9]{1,4}\b", re.NOFLAG),
        pii_type=PIIType.BANK_ACCOUNT.value,
        base_score=0.9,
        validator="iban",
        on_invalid_score=0.5,
        context=("iban", "cuenta", "account", "banco", "bank", "transferencia", "swift"),
    ),
    Rule(
        name="credit_card",
        pattern=_re(r"\b(?:\d{4}[\s\-]?){3}\d{1,4}\b"),
        pii_type=PIIType.CREDIT_CARD.value,
        base_score=0.88,
        validator="luhn",
        on_invalid_score=0.5,
        context=("tarjeta", "card", "visa", "mastercard", "amex", "credit", "débito", "debito"),
    ),
    Rule(
        name="cvv",
        pattern=_re(r"(?:cvv|cvc|cod(?:igo)?\.?\s*seg(?:uridad)?)\s*[:=]?\s*(\d{3,4})\b"),
        pii_type=PIIType.CREDIT_CARD.value,
        base_score=0.9,
        group=1,
    ),
    Rule(
        name="bank_account_labeled",
        pattern=_re(
            r"(?:n[ºo°]?\.?\s*de\s*cuenta|cuenta\s*(?:bancaria|corriente)?|account\s*(?:number|no\.?|#)|routing)"
            r"\s*[:#]?\s*([0-9][0-9\s\-]{6,24}[0-9])"
        ),
        pii_type=PIIType.BANK_ACCOUNT.value,
        base_score=0.85,
        group=1,
    ),
    # ------------------------------------------------------- government IDs
    Rule(
        name="dni_nie",
        pattern=_re(r"\b[XYZ]?\d{7,8}[\s-]?[A-Z]\b", re.NOFLAG),
        pii_type=PIIType.GOV_ID.value,
        base_score=0.85,
        validator="dni",
        on_invalid_score=None,  # without a valid letter it is just a number
        context=("dni", "nie", "identidad", "documento", "identificación", "identificacion"),
        langs=("es",),
    ),
    Rule(
        name="nif_cif",
        pattern=_re(r"\b[ABCDEFGHJNPQRSUVW]\d{7}[0-9A-J]\b", re.NOFLAG),
        pii_type=PIIType.GOV_ID.value,
        base_score=0.85,
        validator="nif",
        on_invalid_score=None,
        context=("nif", "cif", "fiscal", "empresa", "sociedad"),
        langs=("es",),
    ),
    Rule(
        name="cuit_cuil",
        pattern=_re(r"\b\d{2}-\d{8}-\d\b"),
        pii_type=PIIType.GOV_ID.value,
        base_score=0.88,
        validator="cuit",
        on_invalid_score=0.5,
        langs=("es",),
    ),
    Rule(
        name="ssn",
        pattern=_re(r"\b\d{3}-\d{2}-\d{4}\b"),
        pii_type=PIIType.GOV_ID.value,
        base_score=0.9,
        validator="ssn",
        on_invalid_score=0.5,
        langs=("en",),
    ),
    Rule(
        name="passport",
        pattern=_re(
            r"(?:pasaporte|passport)\s*(?:n[ºo°]?\.?|number|no\.?|#)?\s*[:#]?\s*"
            r"([A-Z0-9]{6,12})\b"
        ),
        pii_type=PIIType.GOV_ID.value,
        base_score=0.87,
        group=1,
    ),
    Rule(
        name="driver_license",
        pattern=_re(
            r"(?:licencia\s*(?:de\s*conducir)?|carn[eé]\s*de\s*conducir|driver'?s?\s*licen[sc]e|dl)"
            r"\s*(?:n[ºo°]?\.?|no\.?|#)?\s*[:#]?\s*([A-Z0-9\-]{5,15})\b"
        ),
        pii_type=PIIType.GOV_ID.value,
        base_score=0.85,
        group=1,
    ),
    # ----------------------------------------------------------------- phone
    Rule(
        name="phone_international",
        pattern=_re(r"(?<![\w.])\+\d{1,3}[\s.\-]?(?:\(?\d{1,4}\)?[\s.\-]?){1,5}\d{2,4}(?![\w])"),
        pii_type=PIIType.PHONE.value,
        base_score=0.92,
    ),
    Rule(
        name="phone_es",
        pattern=_re(r"(?<![\w.+])(?:[6789]\d{2}[\s.\-]?\d{2}[\s.\-]?\d{2}[\s.\-]?\d{2}|[6789]\d{2}[\s.\-]?\d{3}[\s.\-]?\d{3})(?![\w])"),
        pii_type=PIIType.PHONE.value,
        base_score=0.8,
        context=("tel", "teléfono", "telefono", "móvil", "movil", "celular", "whatsapp", "contacto", "fax"),
        langs=("es",),
    ),
    Rule(
        name="phone_us",
        pattern=_re(r"(?<![\w.+])\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}(?![\w])"),
        pii_type=PIIType.PHONE.value,
        base_score=0.85,
        langs=("en",),
    ),
    Rule(
        name="phone_labeled",
        pattern=_re(
            r"(?:tel[eé]fono|tel\.?|m[oó]vil|celular|phone|mobile|cell|fax|whatsapp)"
            r"\s*[:#]?\s*((?:\+?\d[\d\s.\-()]{6,20}\d))"
        ),
        pii_type=PIIType.PHONE.value,
        base_score=0.9,
        group=1,
    ),
    # ------------------------------------------------------------- addresses
    Rule(
        name="address_es",
        pattern=_re(
            r"\b(?:calle|c/|avda\.?|avenida|av\.|plaza|pza\.?|paseo|ronda|carrera|carretera|ctra\.?|camino|urbanizaci[oó]n)"
            # Digits are deliberately excluded from the street body: otherwise the
            # greedy body swallows the street number and the floor/door suffix is
            # left dangling outside the match.
            r"\s+[A-Za-zÁÉÍÓÚÑáéíóúñ.'\-]+(?:\s+[A-Za-zÁÉÍÓÚÑáéíóúñ.'\-]+){0,4}"
            r"(?:\s*,?\s*(?:n[ºo°]?\.?\s*)?\d{1,4}[A-Za-z]?\b)?"
            r"(?:\s*,?\s*(?:piso|pta\.?|puerta|esc\.?|bajo|izq\.?|dcha\.?|\d{1,2}\s*[ºª]?)"
            r"(?:\s*[A-Za-z0-9]{1,4}\b)?)?"
        ),
        pii_type=PIIType.ADDRESS.value,
        base_score=0.78,
        langs=("es",),
    ),
    Rule(
        name="address_en",
        pattern=_re(
            r"\b\d{1,5}\s+(?:[A-Z][A-Za-z.'\-]+\s+){1,4}"
            r"(?:street|st\.?|avenue|ave\.?|road|rd\.?|boulevard|blvd\.?|lane|ln\.?|drive|dr\.?|court|ct\.?|way|place|pl\.?|terrace|parkway|circle)"
            r"(?:\s+(?:NW|NE|SW|SE|N|S|E|W)\b)?"
            r"(?:\s*,?\s*(?:apt\.?|suite|ste\.?|unit|#)\s*[A-Za-z0-9\-]+)?"
        ),
        pii_type=PIIType.ADDRESS.value,
        base_score=0.8,
        langs=("en",),
    ),
    Rule(
        name="postal_code",
        pattern=_re(
            r"(?:c[oó]digo\s*postal|cp|zip(?:\s*code)?|postal\s*code)\s*[:#]?\s*(\d{5}(?:-\d{4})?)\b"
        ),
        pii_type=PIIType.ADDRESS.value,
        base_score=0.85,
        group=1,
    ),
    # ----------------------------------------------------------------- dates
    Rule(
        name="date_numeric",
        pattern=_re(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b"),
        pii_type=PIIType.DATE_TIME.value,
        base_score=0.85,
    ),
    Rule(
        name="date_textual",
        pattern=_re(
            rf"\b\d{{1,2}}\s*(?:de\s+)?{_MONTHS}\.?\s*(?:de\s+|,\s*)?\d{{2,4}}\b"
            rf"|\b{_MONTHS}\.?\s+\d{{1,2}},?\s+\d{{4}}\b"
        ),
        pii_type=PIIType.DATE_TIME.value,
        base_score=0.85,
    ),
    # ------------------------------------------------------------ net / user
    Rule(
        name="ipv4",
        pattern=_re(r"\b(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\b"),
        pii_type=PIIType.IP_ADDRESS.value,
        base_score=0.95,
    ),
    Rule(
        name="ipv6",
        pattern=_re(r"\b(?:[A-F0-9]{1,4}:){7}[A-F0-9]{1,4}\b"),
        pii_type=PIIType.IP_ADDRESS.value,
        base_score=0.95,
    ),
    Rule(
        name="mac_address",
        pattern=_re(r"\b(?:[0-9A-F]{2}[:-]){5}[0-9A-F]{2}\b"),
        pii_type=PIIType.IP_ADDRESS.value,
        base_score=0.93,
    ),
    Rule(
        name="username_labeled",
        pattern=_re(
            # The separator is mandatory and one qualifier word may sit before it:
            # with an optional separator, "Usuario deseado: xyz" captured the word
            # "deseado" as the username and redacted the field label itself.
            r"(?:usuario|user(?:name)?|login|nick|alias|cuenta\s*de\s*usuario)"
            r"(?:\s+[A-Za-zÁÉÍÓÚÑáéíóúñ]+)?\s*[:=#]\s*"
            r"([A-Za-z0-9._\-]{3,32})\b"
        ),
        pii_type=PIIType.USERNAME.value,
        base_score=0.82,
        group=1,
    ),
    Rule(
        name="username_handle",
        pattern=_re(r"(?<![\w@])@[A-Za-z0-9._]{3,30}\b"),
        pii_type=PIIType.USERNAME.value,
        base_score=0.8,
    ),
    # ---------------------------------------------------------------- person
    # Names are where a rule engine structurally loses to a learned model: it can
    # only fire on an explicit honorific or an explicit field label.
    Rule(
        name="person_honorific",
        pattern=_re(
            r"\b(?:sr\.?a?\.?|sra\.?|srta\.?|do[nñ]a?|d\.?[ªº]?|mr\.?|mrs\.?|ms\.?|miss|dr\.?a?\.?|ing\.?|lic\.?|prof\.?)"
            r"[ \t]+((?:[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ'\-]+[ \t]*){1,4})"
        ),
        pii_type=PIIType.PERSON.value,
        base_score=0.72,
        group=1,
    ),
    Rule(
        name="person_labeled",
        pattern=_re(
            r"(?:nombre(?:\s*(?:y\s*apellidos?|completo))?|cliente|titular|solicitante|arrendatario|arrendador"
            r"|destinatario|remitente|atenci[oó]n\s*a|a\s*la\s*atenci[oó]n\s*de"
            r"|name|full\s*name|customer|client|tenant|landlord|recipient|attn\.?|bill\s*to|ship\s*to)"
            # A personal name never spans a line break: allowing \s here made the
            # rule swallow the next field's label and redact it away with the name.
            r"[ \t]*[:#][ \t]*((?:[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ'\-]+[ \t]*){1,4})"
        ),
        pii_type=PIIType.PERSON.value,
        base_score=0.75,
        group=1,
    ),
]

_RULES_BY_NAME = {r.name: r for r in RULES}


class RegexRuleDetector:
    """Deterministic PII detector driven by :data:`RULES`."""

    name = "rules"

    def __init__(self, rules: list[Rule] | None = None, context_window: int = CONTEXT_WINDOW):
        self.rules = rules if rules is not None else RULES
        self.context_window = context_window

    # -- context -----------------------------------------------------------
    def _context_hit(
        self,
        text: str,
        start: int,
        end: int,
        keywords: tuple[str, ...],
        same_line: bool = False,
    ) -> str | None:
        """Look for any of ``keywords`` in a window around ``[start, end)``.

        ``same_line`` clamps the window to the current line. Business documents
        are line-oriented -- ``Fecha de nacimiento: ...`` on one line must not
        lend its meaning to ``Fecha de emision: ...`` on the next one.
        """
        if not keywords:
            return None
        lo = max(0, start - self.context_window)
        hi = min(len(text), end + self.context_window)
        if same_line:
            lo = max(lo, text.rfind("\n", 0, start) + 1)
            nl = text.find("\n", end)
            hi = min(hi, nl if nl != -1 else len(text))
        window = text[lo:hi].lower()
        return next((kw for kw in keywords if kw in window), None)

    # -- main --------------------------------------------------------------
    def detect(self, text: str, lang: str = "es") -> list[PIIEntity]:
        found: list[PIIEntity] = []
        for rule in self.rules:
            if lang not in rule.langs:
                continue
            for match in rule.pattern.finditer(text):
                span = match.span(rule.group) if rule.group else match.span()
                if span[0] < 0:
                    continue
                value = text[span[0] : span[1]]
                if rule.guard is not None and not rule.guard(value):
                    continue

                score = rule.base_score
                meta: dict[str, object] = {"rule": rule.name}

                if rule.validator:
                    ok = VALIDATORS[rule.validator](value)
                    meta["checksum"] = rule.validator
                    meta["checksum_passed"] = ok
                    if ok:
                        # A passing checksum is near-proof; cap below 1.0 so the
                        # calibration layer still has room to move.
                        score = min(0.99, score + 0.11)
                    elif rule.on_invalid_score is None:
                        continue
                    else:
                        score = rule.on_invalid_score

                hit = self._context_hit(text, span[0], span[1], rule.context)
                if hit:
                    score = min(0.99, score + 0.06)
                    meta["context_keyword"] = hit
                elif rule.require_context:
                    continue
                elif rule.context:
                    # Rule declared context keywords and found none: the match is
                    # shape-only, so it is weaker than the rule's nominal score.
                    score = max(0.35, score - 0.12)
                    meta["context_keyword"] = None

                # A generic date sitting next to birth-date wording is a DOB.
                pii_type = rule.pii_type
                if pii_type == PIIType.DATE_TIME.value:
                    dob_hit = self._context_hit(
                        text, span[0], span[1], _DOB_CONTEXT, same_line=True
                    )
                    if dob_hit:
                        pii_type = PIIType.DATE_OF_BIRTH.value
                        score = min(0.99, score + 0.05)
                        meta["dob_context"] = dob_hit

                found.append(
                    PIIEntity(
                        text=value,
                        type=pii_type,
                        start=span[0],
                        end=span[1],
                        score=score,
                        source=self.name,
                        metadata=meta,
                    )
                )

        return resolve_overlaps(normalize_entities(text, found))
