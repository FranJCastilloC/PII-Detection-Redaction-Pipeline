# Baseline vs fine-tuned model comparison

> The ai4privacy corpus contains no card or bank-account numbers, so ['BANK_ACCOUNT', 'CREDIT_CARD'] are unreachable for the fine-tuned model by construction and are covered by the checksum-backed rule engine inside the ensemble.

## Test set: `ai4privacy` (500 documents)

| system                      |   P_exact |   R_exact |   F1_exact |   F1_partial |   F1_macro |   entity_leakage |   over_redaction |   docs_per_sec |
|:----------------------------|----------:|----------:|-----------:|-------------:|-----------:|-----------------:|-----------------:|---------------:|
| Rules (regex + checksums)   |    0.7556 |    0.2448 |     0.3698 |       0.3929 |     0.3847 |           0.7287 |           0.0053 |      4340.3900 |
| Presidio                    |    0.3648 |    0.2357 |     0.2864 |       0.3222 |     0.3084 |           0.6299 |           0.0384 |        23.6700 |
| Baseline (rules + Presidio) |    0.4335 |    0.3218 |     0.3694 |       0.4132 |     0.4002 |           0.5489 |           0.0403 |        37.2500 |
| Fine-tuned model            |    0.9190 |    0.9226 |     0.9208 |       0.9516 |     0.8970 |           0.0625 |           0.0033 |        38.1600 |
| Ensemble (baseline + model) |    0.7327 |    0.9012 |     0.8083 |       0.8428 |     0.7586 |           0.0683 |           0.0397 |        21.5400 |

## Test set: `synthetic_documents` (250 documents)

| system                      |   P_exact |   R_exact |   F1_exact |   F1_partial |   F1_macro |   entity_leakage |   over_redaction |   docs_per_sec |
|:----------------------------|----------:|----------:|-----------:|-------------:|-----------:|-----------------:|-----------------:|---------------:|
| Rules (regex + checksums)   |    0.9368 |    0.5609 |     0.7017 |       0.7343 |     0.7314 |           0.4361 |           0.0002 |      1537.3400 |
| Presidio                    |    0.5569 |    0.5722 |     0.5644 |       0.6086 |     0.5291 |           0.3048 |           0.0459 |        30.4300 |
| Baseline (rules + Presidio) |    0.6216 |    0.6857 |     0.6521 |       0.7120 |     0.7130 |           0.2448 |           0.0459 |        30.4000 |
| Fine-tuned model            |    0.5042 |    0.5787 |     0.5389 |       0.6506 |     0.5421 |           0.3148 |           0.0354 |        36.4100 |
| Ensemble (baseline + model) |    0.5989 |    0.8070 |     0.6875 |       0.7650 |     0.7884 |           0.1478 |           0.0711 |        19.0000 |

## Per-type detail — Ensemble (baseline + model) on synthetic documents (partial match, IoU 0.5)

| type          |      P |      R |     F1 |   support |
|:--------------|-------:|-------:|-------:|----------:|
| ADDRESS       | 0.4056 | 0.7900 | 0.5360 |       400 |
| BANK_ACCOUNT  | 0.9934 | 1.0000 | 0.9967 |       150 |
| CREDENTIAL    | 0.9583 | 0.9200 | 0.9388 |       100 |
| CREDIT_CARD   | 1.0000 | 0.8200 | 0.9011 |       100 |
| DATE_OF_BIRTH | 1.0000 | 0.8333 | 0.9091 |       150 |
| DATE_TIME     | 0.5319 | 1.0000 | 0.6944 |       150 |
| EMAIL         | 1.0000 | 1.0000 | 1.0000 |       300 |
| GOV_ID        | 0.4717 | 1.0000 | 0.6410 |       200 |
| IP_ADDRESS    | 1.0000 | 1.0000 | 1.0000 |        50 |
| PERSON        | 0.6207 | 0.8743 | 0.7260 |       350 |
| PHONE         | 0.8981 | 0.7760 | 0.8326 |       250 |
| USERNAME      | 0.9901 | 1.0000 | 0.9950 |       100 |
