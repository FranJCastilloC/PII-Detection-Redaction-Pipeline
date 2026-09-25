# PII Detection & Redaction Pipeline
# Usage: make setup && make all

PY := .venv/bin/python
PYTHONPATH := src
export PYTHONPATH

.PHONY: help setup data train calibrate evaluate examples report test app all clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the venv, install dependencies and spaCy models
	uv venv --python 3.11 .venv
	.venv/bin/uv pip install -r requirements.txt || uv pip install --python .venv -r requirements.txt
	$(PY) -m spacy download en_core_web_md
	$(PY) -m spacy download es_core_news_md

data:  ## Download the corpus, build the splits and generate synthetic documents
	$(PY) scripts/01_build_data.py

train:  ## Fine-tune the token classification model
	$(PY) scripts/02_train.py --limit-train 15000 --limit-val 2000 --epochs 2 --batch-size 16 --max-length 192

calibrate:  ## Fit the confidence calibrator on validation
	$(PY) scripts/03_calibrate.py

evaluate:  ## Compare every system across both test sets
	$(PY) scripts/04_evaluate.py

examples:  ## Run the pipeline over documents and save examples + queue
	$(PY) scripts/06_examples.py

report:  ## Generate tables and figures under results/
	$(PY) scripts/05_report.py

test:  ## Run the test suite
	$(PY) -m pytest

# Each project under Proyectos_Do_it owns a fixed Streamlit port so their
# demos can run side by side. Override with: make app PORT=xxxx
PORT ?= 8501

app:  ## Launch the Streamlit demo (override with PORT=xxxx)
	# `python -m` instead of the console script: venv launchers hard-code the
	# absolute path they were created with, so moving the repo breaks them.
	$(PY) -m streamlit run app/streamlit_app.py --server.port $(PORT)

all: data train calibrate evaluate examples report test  ## Full pipeline

clean:  ## Remove regenerable artefacts
	rm -rf data/raw data/processed models/distilbert-pii-ner data/*.sqlite data/*.jsonl
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
