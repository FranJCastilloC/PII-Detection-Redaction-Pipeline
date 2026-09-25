"""Checksum validators: the guarantees the confidence scores lean on."""

import pytest

from pii_pipeline.detectors.validators import (
    cuit_check, dni_check, iban_check, luhn_check, nif_check, ssn_check,
)


@pytest.mark.parametrize("value", ["4539 1488 0343 6467", "4539148803436467", "5425-2334-3010-9903"])
def test_luhn_accepts_valid_cards(value):
    assert luhn_check(value)


@pytest.mark.parametrize("value", ["4539 1488 0343 6461", "1234567812345678", "42", ""])
def test_luhn_rejects_invalid_cards(value):
    assert not luhn_check(value)


@pytest.mark.parametrize("value", ["ES91 2100 0418 4502 0005 1332", "GB82WEST12345698765432"])
def test_iban_accepts_valid(value):
    assert iban_check(value)


@pytest.mark.parametrize("value", ["ES91 2100 0418 4502 0005 1333", "ZZ00 1234", "not an iban"])
def test_iban_rejects_invalid(value):
    assert not iban_check(value)


@pytest.mark.parametrize("value", ["12345678Z", "X1234567L"])
def test_dni_accepts_valid(value):
    assert dni_check(value)


def test_dni_rejects_wrong_control_letter():
    assert not dni_check("12345678A")


def test_cuit_and_ssn():
    assert cuit_check("20-12345678-6")
    assert not cuit_check("20-12345678-5")
    assert ssn_check("123-45-6789")
    # 666 and 9xx are never issued as SSN area numbers.
    assert not ssn_check("666-45-6789")


def test_nif_accepts_valid_entity_id():
    assert nif_check("A58818501")
