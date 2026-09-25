"""Synthetic business documents with perfect ground-truth PII offsets.

The ai4privacy corpus is made of short, isolated sentences. Real redaction work
happens on whole documents, where layout, field labels and repeated mentions all
change the problem. This module generates full invoices, contracts, emails,
support tickets and onboarding forms with Faker, in Spanish and English.

The trick that makes it a usable test set: the document is assembled through a
builder that records the character span of every PII value *as it is written*,
so the annotation is exact by construction and never needs manual labelling.

Documents also contain deliberate **distractors** -- invoice numbers, order refs,
amounts and product codes that look like identifiers but are not PII. They are
left unannotated so the evaluation can measure over-redaction honestly.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from faker import Faker

from ..entities import PIIType

DOC_TYPES = ("invoice", "contract", "email", "support_ticket", "onboarding_form")


@dataclass
class DocumentBuilder:
    """Assembles text while recording the offsets of every PII value written."""

    parts: list[str] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    _len: int = 0

    def w(self, text: str) -> "DocumentBuilder":
        """Write plain, non-PII text."""
        self.parts.append(text)
        self._len += len(text)
        return self

    def p(self, value: str, pii_type: str) -> "DocumentBuilder":
        """Write a PII value and record its exact span."""
        value = str(value)
        self.entities.append(
            {
                "start": self._len,
                "end": self._len + len(value),
                "type": pii_type,
                "text": value,
            }
        )
        return self.w(value)

    def nl(self, count: int = 1) -> "DocumentBuilder":
        return self.w("\n" * count)

    def build(self) -> tuple[str, list[dict[str, Any]]]:
        text = "".join(self.parts)
        # Cheap invariant check: a mis-recorded offset would poison every metric
        # computed downstream, so fail loudly here instead.
        for ent in self.entities:
            assert text[ent["start"] : ent["end"]] == ent["text"], (
                f"offset drift for {ent['text']!r}"
            )
        return text, self.entities


class PersonaFactory:
    """A coherent fake person: the same human across a whole document."""

    def __init__(self, fake: Faker, lang: str, rng: random.Random):
        self.fake = fake
        self.lang = lang
        self.rng = rng

    def make(self) -> dict[str, str]:
        fake, rng = self.fake, self.rng
        first, last = fake.first_name(), fake.last_name()
        full = f"{first} {last}"
        user = f"{first[0]}{last}".lower().replace(" ", "")[:14]
        person = {
            "full_name": full,
            "first_name": first,
            "last_name": last,
            "email": f"{user}@{fake.domain_name()}",
            "phone": fake.phone_number(),
            "address": fake.street_address().replace("\n", ", "),
            "city": fake.city(),
            "postcode": fake.postcode(),
            "dob": fake.date_of_birth(minimum_age=21, maximum_age=78).strftime(
                "%d/%m/%Y" if self.lang == "es" else "%m/%d/%Y"
            ),
            "iban": fake.iban(),
            "card": fake.credit_card_number(),
            "username": user + str(rng.randint(10, 99)),
            "password": fake.password(length=12),
            "ip": fake.ipv4(),
            "company": fake.company(),
        }
        person["gov_id"] = self._gov_id()
        return person

    def _gov_id(self) -> str:
        """A locale-appropriate national identifier."""
        for attr in ("nif", "ssn", "doi"):
            generator = getattr(self.fake, attr, None)
            if callable(generator):
                try:
                    return str(generator())
                except Exception:  # pragma: no cover - provider quirks
                    continue
        return str(self.rng.randint(10_000_000, 99_999_999))


def _distractors(fake: Faker, rng: random.Random) -> dict[str, str]:
    """Identifier-shaped strings that are NOT PII, to measure over-redaction."""
    return {
        "invoice_no": f"{rng.randint(2022, 2025)}-{rng.randint(1000, 9999)}",
        "order_ref": f"ORD-{rng.randint(100000, 999999)}",
        "sku": f"SKU-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}",
        "ticket_no": f"TCK-{rng.randint(10000, 99999)}",
        "amount": f"{rng.randint(120, 9800)},{rng.randint(10, 99)}",
        "vat": str(rng.choice([10, 21])),
        "contract_no": f"CT-{rng.randint(2022, 2025)}-{rng.randint(100, 999)}",
    }


# ---------------------------------------------------------------------------
# Document templates
# ---------------------------------------------------------------------------
def _invoice(b: DocumentBuilder, p: dict, d: dict, fake: Faker, lang: str, rng) -> None:
    if lang == "es":
        b.w(f"FACTURA N.º {d['invoice_no']}\n")
        b.w(f"{p['company']}\nFecha de emisión: ").p(
            fake.date_this_year().strftime("%d/%m/%Y"), PIIType.DATE_TIME.value
        ).nl(2)
        b.w("DATOS DEL CLIENTE\n")
        b.w("Nombre: ").p(p["full_name"], PIIType.PERSON.value).nl()
        b.w("NIF/DNI: ").p(p["gov_id"], PIIType.GOV_ID.value).nl()
        b.w("Domicilio: ").p(p["address"], PIIType.ADDRESS.value).w(", ").p(
            p["city"], PIIType.ADDRESS.value
        ).w(" ").p(p["postcode"], PIIType.ADDRESS.value).nl()
        b.w("Teléfono: ").p(p["phone"], PIIType.PHONE.value).w("   Email: ").p(
            p["email"], PIIType.EMAIL.value
        ).nl(2)
        b.w(f"Concepto: Servicios de consultoría (ref. {d['order_ref']}, {d['sku']})\n")
        b.w(f"Base imponible: {d['amount']} EUR   IVA ({d['vat']}%)\n\n")
        b.w("FORMA DE PAGO\n")
        b.w("Domiciliación en cuenta ").p(p["iban"], PIIType.BANK_ACCOUNT.value).nl()
        b.w("Tarjeta asociada: ").p(p["card"], PIIType.CREDIT_CARD.value).nl()
    else:
        b.w(f"INVOICE #{d['invoice_no']}\n")
        b.w(f"{p['company']}\nIssue date: ").p(
            fake.date_this_year().strftime("%m/%d/%Y"), PIIType.DATE_TIME.value
        ).nl(2)
        b.w("BILL TO\n")
        b.w("Name: ").p(p["full_name"], PIIType.PERSON.value).nl()
        b.w("Tax ID: ").p(p["gov_id"], PIIType.GOV_ID.value).nl()
        b.w("Address: ").p(p["address"], PIIType.ADDRESS.value).w(", ").p(
            p["city"], PIIType.ADDRESS.value
        ).w(" ").p(p["postcode"], PIIType.ADDRESS.value).nl()
        b.w("Phone: ").p(p["phone"], PIIType.PHONE.value).w("   Email: ").p(
            p["email"], PIIType.EMAIL.value
        ).nl(2)
        b.w(f"Description: Consulting services (ref. {d['order_ref']}, {d['sku']})\n")
        b.w(f"Subtotal: {d['amount']} USD   Tax ({d['vat']}%)\n\n")
        b.w("PAYMENT\n")
        b.w("Bank account ").p(p["iban"], PIIType.BANK_ACCOUNT.value).nl()
        b.w("Card on file: ").p(p["card"], PIIType.CREDIT_CARD.value).nl()


def _contract(b: DocumentBuilder, p: dict, d: dict, fake: Faker, lang: str, rng) -> None:
    other = PersonaFactory(fake, lang, rng).make()
    if lang == "es":
        b.w(f"CONTRATO DE PRESTACIÓN DE SERVICIOS N.º {d['contract_no']}\n\n")
        b.w("REUNIDOS\n\nDe una parte, D./Dña. ").p(
            p["full_name"], PIIType.PERSON.value
        ).w(", mayor de edad, con DNI ").p(p["gov_id"], PIIType.GOV_ID.value)
        b.w(", nacido/a el ").p(p["dob"], PIIType.DATE_OF_BIRTH.value)
        b.w(", y domicilio en ").p(p["address"], PIIType.ADDRESS.value).w(", ").p(
            p["city"], PIIType.ADDRESS.value
        ).w(" (en adelante, el ARRENDATARIO).\n\n")
        b.w("Y de otra parte, D./Dña. ").p(other["full_name"], PIIType.PERSON.value)
        b.w(", con domicilio en ").p(other["address"], PIIType.ADDRESS.value)
        b.w(" y correo electrónico ").p(other["email"], PIIType.EMAIL.value)
        b.w(" (en adelante, el ARRENDADOR).\n\n")
        b.w("CLÁUSULAS\n\nPRIMERA.- El precio se fija en ")
        b.w(f"{d['amount']} EUR anuales.\n")
        b.w("SEGUNDA.- El pago se domiciliará en la cuenta ").p(
            p["iban"], PIIType.BANK_ACCOUNT.value
        ).w(".\n")
        b.w("TERCERA.- Las notificaciones se dirigirán al teléfono ").p(
            p["phone"], PIIType.PHONE.value
        ).w(".\n\n")
        b.w("Y en prueba de conformidad, firman en ").p(
            fake.city(), PIIType.ADDRESS.value
        ).w(" a ").p(
            fake.date_this_year().strftime("%d/%m/%Y"), PIIType.DATE_TIME.value
        ).w(".\n")
    else:
        b.w(f"SERVICES AGREEMENT No. {d['contract_no']}\n\n")
        b.w("BETWEEN\n\nParty A: ").p(p["full_name"], PIIType.PERSON.value)
        b.w(", holder of identification number ").p(p["gov_id"], PIIType.GOV_ID.value)
        b.w(", born on ").p(p["dob"], PIIType.DATE_OF_BIRTH.value)
        b.w(", residing at ").p(p["address"], PIIType.ADDRESS.value).w(", ").p(
            p["city"], PIIType.ADDRESS.value
        ).w(" (the \"Client\").\n\n")
        b.w("Party B: ").p(other["full_name"], PIIType.PERSON.value)
        b.w(", of ").p(other["address"], PIIType.ADDRESS.value)
        b.w(", reachable at ").p(other["email"], PIIType.EMAIL.value)
        b.w(" (the \"Provider\").\n\n")
        b.w(f"TERMS\n\n1. The annual fee is set at {d['amount']} USD.\n")
        b.w("2. Payment shall be made to account ").p(
            p["iban"], PIIType.BANK_ACCOUNT.value
        ).w(".\n")
        b.w("3. Notices shall be sent to ").p(p["phone"], PIIType.PHONE.value).w(".\n\n")
        b.w("Signed in ").p(fake.city(), PIIType.ADDRESS.value).w(" on ").p(
            fake.date_this_year().strftime("%m/%d/%Y"), PIIType.DATE_TIME.value
        ).w(".\n")


def _email(b: DocumentBuilder, p: dict, d: dict, fake: Faker, lang: str, rng) -> None:
    agent = PersonaFactory(fake, lang, rng).make()
    if lang == "es":
        b.w("De: ").p(agent["email"], PIIType.EMAIL.value).nl()
        b.w("Para: ").p(p["email"], PIIType.EMAIL.value).nl()
        b.w(f"Asunto: Confirmación de su pedido {d['order_ref']}\n\n")
        b.w("Estimado/a ").p(p["full_name"], PIIType.PERSON.value).w(":\n\n")
        b.w(f"Le confirmamos que su pedido {d['order_ref']} ha sido procesado. ")
        b.w("El envío se realizará a ").p(p["address"], PIIType.ADDRESS.value).w(", ").p(
            p["city"], PIIType.ADDRESS.value
        ).w(".\n\n")
        b.w("Hemos registrado el cargo en la tarjeta terminada en ").p(
            p["card"], PIIType.CREDIT_CARD.value
        ).w(f" por importe de {d['amount']} EUR.\n\n")
        b.w("Si necesita modificar el envío, llámenos al ").p(
            p["phone"], PIIType.PHONE.value
        ).w(" indicando su número de cliente.\n\n")
        b.w("Un cordial saludo,\n").p(agent["full_name"], PIIType.PERSON.value).nl()
        b.w(f"{agent['company']} - Atención al cliente\n")
    else:
        b.w("From: ").p(agent["email"], PIIType.EMAIL.value).nl()
        b.w("To: ").p(p["email"], PIIType.EMAIL.value).nl()
        b.w(f"Subject: Your order {d['order_ref']} is confirmed\n\n")
        b.w("Dear ").p(p["full_name"], PIIType.PERSON.value).w(",\n\n")
        b.w(f"We are writing to confirm that order {d['order_ref']} has been processed. ")
        b.w("It will be shipped to ").p(p["address"], PIIType.ADDRESS.value).w(", ").p(
            p["city"], PIIType.ADDRESS.value
        ).w(".\n\n")
        b.w("We charged the card ending in ").p(p["card"], PIIType.CREDIT_CARD.value)
        b.w(f" for {d['amount']} USD.\n\n")
        b.w("To change the delivery address, call us at ").p(
            p["phone"], PIIType.PHONE.value
        ).w(" with your customer number at hand.\n\n")
        b.w("Best regards,\n").p(agent["full_name"], PIIType.PERSON.value).nl()
        b.w(f"{agent['company']} - Customer Care\n")


def _support_ticket(b: DocumentBuilder, p: dict, d: dict, fake: Faker, lang: str, rng) -> None:
    if lang == "es":
        b.w(f"TICKET DE SOPORTE {d['ticket_no']}\n")
        b.w(f"Prioridad: {rng.choice(['Alta', 'Media', 'Baja'])}   Estado: Abierto\n\n")
        b.w("Reportado por: ").p(p["full_name"], PIIType.PERSON.value).w(" (").p(
            p["email"], PIIType.EMAIL.value
        ).w(")\n")
        b.w("Usuario de la plataforma: ").p(p["username"], PIIType.USERNAME.value).nl()
        b.w("Teléfono de contacto: ").p(p["phone"], PIIType.PHONE.value).nl(2)
        b.w("DESCRIPCIÓN\nEl cliente no puede acceder a su cuenta. ")
        b.w("Indica que restableció la contraseña a ").p(
            p["password"], PIIType.CREDENTIAL.value
        ).w(" y aun así recibe un error.\n")
        b.w("Último acceso registrado desde la IP ").p(p["ip"], PIIType.IP_ADDRESS.value)
        b.w(" el ").p(
            fake.date_this_month().strftime("%d/%m/%Y"), PIIType.DATE_TIME.value
        ).w(".\n\n")
        b.w("VERIFICACIÓN DE IDENTIDAD\n")
        b.w("Documento aportado: ").p(p["gov_id"], PIIType.GOV_ID.value)
        b.w("   Fecha de nacimiento: ").p(p["dob"], PIIType.DATE_OF_BIRTH.value).nl()
        b.w(f"Nota interna: revisar duplicidad con {d['ticket_no']}.\n")
    else:
        b.w(f"SUPPORT TICKET {d['ticket_no']}\n")
        b.w(f"Priority: {rng.choice(['High', 'Medium', 'Low'])}   Status: Open\n\n")
        b.w("Reported by: ").p(p["full_name"], PIIType.PERSON.value).w(" (").p(
            p["email"], PIIType.EMAIL.value
        ).w(")\n")
        b.w("Platform username: ").p(p["username"], PIIType.USERNAME.value).nl()
        b.w("Contact phone: ").p(p["phone"], PIIType.PHONE.value).nl(2)
        b.w("DESCRIPTION\nThe customer cannot sign in. ")
        b.w("They report resetting the password to ").p(
            p["password"], PIIType.CREDENTIAL.value
        ).w(" and still getting an error.\n")
        b.w("Last recorded login from IP ").p(p["ip"], PIIType.IP_ADDRESS.value)
        b.w(" on ").p(
            fake.date_this_month().strftime("%m/%d/%Y"), PIIType.DATE_TIME.value
        ).w(".\n\n")
        b.w("IDENTITY VERIFICATION\n")
        b.w("Document provided: ").p(p["gov_id"], PIIType.GOV_ID.value)
        b.w("   Date of birth: ").p(p["dob"], PIIType.DATE_OF_BIRTH.value).nl()
        b.w(f"Internal note: check for duplicate of {d['ticket_no']}.\n")


def _onboarding_form(b: DocumentBuilder, p: dict, d: dict, fake: Faker, lang: str, rng) -> None:
    if lang == "es":
        b.w("FORMULARIO DE ALTA DE CLIENTE\n")
        b.w(f"Referencia interna: {d['order_ref']}\n\n")
        b.w("Nombre y apellidos: ").p(p["full_name"], PIIType.PERSON.value).nl()
        b.w("Fecha de nacimiento: ").p(p["dob"], PIIType.DATE_OF_BIRTH.value).nl()
        b.w("Documento de identidad: ").p(p["gov_id"], PIIType.GOV_ID.value).nl()
        b.w("Dirección: ").p(p["address"], PIIType.ADDRESS.value).nl()
        b.w("Ciudad: ").p(p["city"], PIIType.ADDRESS.value).w("    Código postal: ").p(
            p["postcode"], PIIType.ADDRESS.value
        ).nl()
        b.w("Teléfono móvil: ").p(p["phone"], PIIType.PHONE.value).nl()
        b.w("Correo electrónico: ").p(p["email"], PIIType.EMAIL.value).nl()
        b.w("Cuenta para domiciliación: ").p(p["iban"], PIIType.BANK_ACCOUNT.value).nl()
        b.w("Usuario deseado: ").p(p["username"], PIIType.USERNAME.value).nl()
        b.w("Contraseña provisional: ").p(p["password"], PIIType.CREDENTIAL.value).nl(2)
        b.w("Declaro que los datos son veraces y autorizo su tratamiento conforme al RGPD.\n")
    else:
        b.w("CUSTOMER ONBOARDING FORM\n")
        b.w(f"Internal reference: {d['order_ref']}\n\n")
        b.w("Full name: ").p(p["full_name"], PIIType.PERSON.value).nl()
        b.w("Date of birth: ").p(p["dob"], PIIType.DATE_OF_BIRTH.value).nl()
        b.w("Identity document: ").p(p["gov_id"], PIIType.GOV_ID.value).nl()
        b.w("Address: ").p(p["address"], PIIType.ADDRESS.value).nl()
        b.w("City: ").p(p["city"], PIIType.ADDRESS.value).w("    ZIP code: ").p(
            p["postcode"], PIIType.ADDRESS.value
        ).nl()
        b.w("Mobile phone: ").p(p["phone"], PIIType.PHONE.value).nl()
        b.w("Email address: ").p(p["email"], PIIType.EMAIL.value).nl()
        b.w("Direct debit account: ").p(p["iban"], PIIType.BANK_ACCOUNT.value).nl()
        b.w("Requested username: ").p(p["username"], PIIType.USERNAME.value).nl()
        b.w("Temporary password: ").p(p["password"], PIIType.CREDENTIAL.value).nl(2)
        b.w("I certify the information above is accurate and consent to its processing.\n")


TEMPLATES: dict[str, Callable[..., None]] = {
    "invoice": _invoice,
    "contract": _contract,
    "email": _email,
    "support_ticket": _support_ticket,
    "onboarding_form": _onboarding_form,
}

LOCALES = {"es": ["es_ES", "es_MX"], "en": ["en_US", "en_GB"]}


def generate(n: int = 250, seed: int = 7) -> list[dict[str, Any]]:
    """Generate ``n`` documents evenly spread over type x language."""
    rng = random.Random(seed)
    Faker.seed(seed)
    fakers = {
        (lang, loc): Faker(loc) for lang, locs in LOCALES.items() for loc in locs
    }
    docs: list[dict[str, Any]] = []
    combos = [(t, l) for t in DOC_TYPES for l in ("es", "en")]
    for i in range(n):
        doc_type, lang = combos[i % len(combos)]
        locale = rng.choice(LOCALES[lang])
        fake = fakers[(lang, locale)]
        persona = PersonaFactory(fake, lang, rng).make()
        distract = _distractors(fake, rng)
        builder = DocumentBuilder()
        TEMPLATES[doc_type](builder, persona, distract, fake, lang, rng)
        text, entities = builder.build()
        docs.append(
            {
                "id": f"synth-{doc_type}-{lang}-{i:04d}",
                "lang": lang,
                "doc_type": doc_type,
                "locale": locale,
                "text": text,
                "entities": entities,
            }
        )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic/documents.jsonl"))
    parser.add_argument("-n", "--num-docs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    docs = generate(args.num_docs, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    from collections import Counter

    types = Counter(e["type"] for d in docs for e in d["entities"])
    print(f"{len(docs)} documents -> {args.out}")
    print(f"{sum(types.values())} entities: {dict(types.most_common())}")


if __name__ == "__main__":
    main()
