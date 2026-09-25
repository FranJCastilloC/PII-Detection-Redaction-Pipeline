"""Publish the fine-tuned PII model to the Hugging Face Hub.

The weights are ~500 MB, so they stay out of git. Hosting them on the Hub is
what lets the deployed Streamlit demo run the full ensemble instead of silently
degrading to the rule baseline.

Authentication is never handled here: log in first with

    .venv/bin/huggingface-cli login

so the token lives in your own HF cache and is never passed through this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_DIR = Path("models/distilbert-pii-ner/final")
UPLOAD_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]


def build_model_card(repo_id: str, training_report: dict, evaluation: dict) -> str:
    """Render a model card carrying the real measured numbers."""
    val = training_report["final_validation"]
    per_type = {
        k: v for k, v in training_report.get("per_type_validation", {}).items()
        if isinstance(v, dict) and "f1-score" in v and "avg" not in k
    }
    rows = "\n".join(
        f"| `{k}` | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1-score']:.3f} | {v['support']} |"
        for k, v in sorted(per_type.items(), key=lambda x: -x[1]["f1-score"])
    )

    ood = evaluation["test_sets"]["synthetic_documents"]["systems"]
    model_f1 = ood["finetuned_model"]["partial"]["micro"]["f1"]
    ens_f1 = ood["ensemble"]["partial"]["micro"]["f1"]
    base_f1 = ood["baseline_rules_presidio"]["partial"]["micro"]["f1"]

    return f"""---
language:
  - es
  - en
license: mit
library_name: transformers
pipeline_tag: token-classification
base_model: distilbert-base-multilingual-cased
tags:
  - pii
  - ner
  - privacy
  - redaction
  - token-classification
---

# DistilBERT multilingual — PII token classification (ES/EN)

Fine-tuned `distilbert-base-multilingual-cased` for detecting personally
identifiable information in Spanish and English business documents. It is the
learned component of the
[PII Detection & Redaction Pipeline](https://github.com/FranJCastilloC/PII-Detection-Redaction-Pipeline),
where it is combined with a checksum-backed rule engine and Presidio.

## Entity types

10 BIO classes: `PERSON`, `EMAIL`, `PHONE`, `ADDRESS`, `DATE_OF_BIRTH`,
`GOV_ID`, `USERNAME`, `CREDENTIAL`, `IP_ADDRESS`, `DATE_TIME`.

**`CREDIT_CARD` and `BANK_ACCOUNT` are deliberately absent.** The training
corpus annotates a card *issuer* but contains no card numbers and no bank
account numbers at all, so the model cannot learn them. In the parent pipeline
those two classes are handled by the rule engine, where Luhn and the IBAN
mod-97 checksum are arithmetic proofs rather than estimates.

## Results

Validation (seqeval, entity level, 2,000 held-out samples):

| | Precision | Recall | F1 |
|---|---|---|---|
| **micro** | {val['eval_precision']:.4f} | {val['eval_recall']:.4f} | **{val['eval_f1']:.4f}** |

| Entity | Precision | Recall | F1 | Support |
|---|---|---|---|---|
{rows}

On **complete business documents** (out-of-distribution, 250 synthetic invoices,
contracts, emails, tickets and forms) F1 drops to **{model_f1:.3f}**, while the
rule baseline reaches {base_f1:.3f} and the full ensemble {ens_f1:.3f}. Formal
documents carry explicit field labels (`Phone:`, `Name:`) that regexes handle
well and that the sentence-level training corpus never contained. Use this model
as one component of an ensemble, not on its own.

## Usage

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForTokenClassification.from_pretrained("{repo_id}")
```

For span decoding, sliding windows over long documents, per-span confidence and
redaction, use the pipeline in the linked repository.

## Training data

[`ai4privacy/pii-masking-300k`](https://huggingface.co/datasets/ai4privacy/pii-masking-300k),
filtered to English and Spanish. The corpus is fully synthetic: **no real
person's data was used at any point.** Character offsets were re-aligned to this
model's own tokenisation rather than reusing the corpus's precomputed BIO tags.

Config: {training_report['n_train_windows']:,} training windows, batch
{training_report['batch_size']}, lr {training_report['learning_rate']},
max length {training_report['max_length']}, 1 epoch.

## Limitations

- Spanish and English only.
- Trained on synthetic sentences; real documents bring OCR noise, abbreviations
  and inconsistent layout that this evaluation does not capture.
- Degrades on document-style input relative to sentence-style input (see above).
- Should not be the only safeguard in a redaction system. The parent pipeline
  pairs it with deterministic rules, calibrated confidence scores and a human
  review queue for exactly this reason.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="e.g. your-user/distilbert-pii-ner-es-en")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--private", action="store_true", help="create the repo as private")
    parser.add_argument("--dry-run", action="store_true", help="print the model card and exit")
    args = parser.parse_args()

    report = json.loads(Path("models/distilbert-pii-ner/training_report.json").read_text())
    evaluation = json.loads(Path("results/metrics/evaluation.json").read_text())
    card = build_model_card(args.repo_id, report, evaluation)

    if args.dry_run:
        print(card)
        return

    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()  # fails loudly if not logged in
    print(f"authenticated as {who.get('name')}")

    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    card_path = args.model_dir / "README.md"
    card_path.write_text(card, encoding="utf-8")

    for name in UPLOAD_FILES + ["README.md"]:
        path = args.model_dir / name
        if not path.exists():
            print(f"  skipping missing {name}")
            continue
        size_mb = path.stat().st_size / 1_048_576
        print(f"  uploading {name} ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=args.repo_id,
            repo_type="model",
        )

    print(f"\ndone -> https://huggingface.co/{args.repo_id}")
    print(f"Set PII_MODEL_REPO={args.repo_id} in the Streamlit app secrets.")


if __name__ == "__main__":
    main()
