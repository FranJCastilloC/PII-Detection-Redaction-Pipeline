"""Checksum validators for structured identifiers.

These are what separate a regex that *looks* like a bank account from one that
provably is. A passing checksum lets the rule engine claim near-certainty; a
failing one is the single most useful signal for routing a span to the human
review queue instead of dropping or auto-redacting it.
"""

from __future__ import annotations

import re

_DIGITS = re.compile(r"\D")


def luhn_check(value: str) -> bool:
    """Validate a credit-card number with the Luhn (mod-10) algorithm."""
    digits = _DIGITS.sub("", value)
    if not 12 <= len(digits) <= 19:
        return False
    total, parity = 0, len(digits) % 2
    for idx, ch in enumerate(digits):
        d = int(ch)
        if idx % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def iban_check(value: str) -> bool:
    """Validate an IBAN with the ISO 13616 mod-97 checksum."""
    cleaned = re.sub(r"[\s-]", "", value).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", cleaned):
        return False
    rearranged = cleaned[4:] + cleaned[:4]
    try:
        numeric = "".join(
            str(int(ch, 36)) if ch.isalpha() else ch for ch in rearranged
        )
    except ValueError:
        return False
    return int(numeric) % 97 == 1


_DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"


def dni_check(value: str) -> bool:
    """Validate a Spanish DNI or NIE control letter."""
    cleaned = re.sub(r"[\s-]", "", value).upper()
    match = re.fullmatch(r"([XYZ]?)(\d{7,8})([A-Z])", cleaned)
    if not match:
        return False
    prefix, number, letter = match.groups()
    # NIE prefixes map onto a leading digit before running the DNI algorithm.
    number = {"X": "0", "Y": "1", "Z": "2"}.get(prefix, "") + number
    try:
        return _DNI_LETTERS[int(number) % 23] == letter
    except (ValueError, IndexError):
        return False


_CUIT_WEIGHTS = (5, 4, 3, 2, 7, 6, 5, 4, 3, 2)


def cuit_check(value: str) -> bool:
    """Validate an Argentine CUIT/CUIL with its mod-11 check digit."""
    digits = _DIGITS.sub("", value)
    if len(digits) != 11:
        return False
    total = sum(w * int(d) for w, d in zip(_CUIT_WEIGHTS, digits[:10]))
    remainder = 11 - (total % 11)
    expected = {11: 0, 10: 9}.get(remainder, remainder)
    return expected == int(digits[10])


def ssn_check(value: str) -> bool:
    """Structural validity of a US SSN (there is no checksum, only dead ranges)."""
    digits = _DIGITS.sub("", value)
    if len(digits) != 9:
        return False
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    if area in {"000", "666"} or area.startswith("9"):
        return False
    return group != "00" and serial != "0000"


def nif_check(value: str) -> bool:
    """Validate a Spanish CIF/NIF for legal entities (letter + 7 digits + control)."""
    cleaned = re.sub(r"[\s-]", "", value).upper()
    match = re.fullmatch(r"([ABCDEFGHJNPQRSUVW])(\d{7})([0-9A-J])", cleaned)
    if not match:
        return False
    _, digits, control = match.groups()
    even = sum(int(d) for d in digits[1::2])
    odd = 0
    for d in digits[0::2]:
        doubled = int(d) * 2
        odd += doubled - 9 if doubled > 9 else doubled
    check = (10 - (even + odd) % 10) % 10
    return control == str(check) or control == "JABCDEFGHI"[check]


VALIDATORS = {
    "luhn": luhn_check,
    "iban": iban_check,
    "dni": dni_check,
    "cuit": cuit_check,
    "ssn": ssn_check,
    "nif": nif_check,
}
