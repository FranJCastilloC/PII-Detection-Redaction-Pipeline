"""The synthetic corpus is a test set: its annotations must be exact."""

from pii_pipeline.data.synth_documents import DOC_TYPES, generate


def test_generated_offsets_match_the_text_exactly():
    for doc in generate(n=30, seed=99):
        for ent in doc["entities"]:
            assert doc["text"][ent["start"] : ent["end"]] == ent["text"]


def test_covers_every_document_type_in_both_languages():
    docs = generate(n=40, seed=5)
    combos = {(d["doc_type"], d["lang"]) for d in docs}
    for doc_type in DOC_TYPES:
        assert (doc_type, "es") in combos
        assert (doc_type, "en") in combos


def test_distractors_are_not_annotated_as_pii():
    # Invoice/order references look like identifiers but must stay unlabelled,
    # otherwise over-redaction cannot be measured.
    for doc in generate(n=20, seed=3):
        annotated = {e["text"] for e in doc["entities"]}
        for token in doc["text"].split():
            if token.startswith(("ORD-", "SKU-", "TCK-", "CT-")):
                assert token.rstrip(".,)") not in annotated
