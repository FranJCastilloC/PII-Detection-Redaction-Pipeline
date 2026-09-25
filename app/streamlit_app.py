"""Interactive demo: paste or upload a document, see what gets redacted and why."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pii_pipeline.entities import TYPE_COLORS  # noqa: E402
from pii_pipeline.pipeline import PIIPipeline, detect_language  # noqa: E402
from pii_pipeline.review_queue import Decision, ReviewPolicy, ReviewStatus, ReviewQueue  # noqa: E402
from pii_pipeline.scoring import ConfidenceCalibrator  # noqa: E402

MODEL_DIR = ROOT / "models/distilbert-pii-ner/final"
CALIBRATOR = ROOT / "models/calibrator.json"
EXAMPLES_DIR = ROOT / "results/examples"
QUEUE_DB = ROOT / "data/review_queue_app.sqlite"

DECISION_STYLE = {
    Decision.AUTO_REDACT.value: ("Automatic", "#1baf7a"),
    Decision.REVIEW.value: ("Review", "#eb6834"),
    Decision.DISCARD.value: ("Discarded", "#888888"),
}

st.set_page_config(page_title="PII Redaction Pipeline", page_icon="🔒", layout="wide")


@st.cache_resource(show_spinner="Loading detectors...")
def load_pipeline(engines: tuple[str, ...]) -> PIIPipeline:
    return PIIPipeline(
        engines=engines,
        model_dir=MODEL_DIR,
        calibrator=ConfidenceCalibrator.load(CALIBRATOR),
        queue=ReviewQueue(QUEUE_DB),
    )


@st.cache_data
def load_examples() -> dict[str, str]:
    if not EXAMPLES_DIR.exists():
        return {}
    return {
        p.stem.replace(".input", "").replace("_", " ").title(): p.read_text(encoding="utf-8")
        for p in sorted(EXAMPLES_DIR.glob("*.input.txt"))
    }


def highlight(text: str, entities, decisions: dict[str, str]) -> str:
    """Render the source text with every detected span highlighted by type."""
    parts, cursor = [], 0
    for ent in sorted(entities, key=lambda e: e.start):
        if ent.start < cursor:
            continue
        parts.append(html.escape(text[cursor : ent.start]))
        color = TYPE_COLORS.get(ent.type, "#888888")
        decision = decisions.get(f"{ent.start}:{ent.end}:{ent.type}", "")
        dashed = "border-bottom:2px dashed #eb6834;" if decision == Decision.REVIEW.value else ""
        parts.append(
            f'<span title="{ent.type} · {ent.score:.2f} · {decision}" '
            f'style="background:{color}2e;border-radius:3px;padding:1px 3px;{dashed}">'
            f"{html.escape(ent.text)}"
            f'<sub style="color:{color};font-size:0.62em;font-weight:600;"> {ent.type}</sub>'
            "</span>"
        )
        cursor = ent.end
    parts.append(html.escape(text[cursor:]))
    body = "".join(parts).replace("\n", "<br>")
    return (
        '<div style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;'
        'font-size:0.82rem;line-height:1.75;white-space:pre-wrap;border:1px solid #e2e2e2;'
        'border-radius:8px;padding:14px;max-height:520px;overflow:auto;">'
        f"{body}</div>"
    )


def plain_block(text: str) -> str:
    return (
        '<div style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;'
        'font-size:0.82rem;line-height:1.75;white-space:pre-wrap;border:1px solid #e2e2e2;'
        'border-radius:8px;padding:14px;max-height:520px;overflow:auto;">'
        f"{html.escape(text).replace(chr(10), '<br>')}</div>"
    )


# --------------------------------------------------------------------------- UI
st.title("PII Detection & Redaction Pipeline")
st.caption(
    "Detects personal data in business documents (ES/EN), redacts it, scores every "
    "detection, and routes the ambiguous cases to a human review queue."
)

with st.sidebar:
    st.header("Settings")

    model_ready = (MODEL_DIR / "config.json").exists()
    engine_choice = st.radio(
        "Detection engine",
        ["Ensemble (rules + Presidio + model)", "Baseline only (rules + Presidio)", "Rules only"],
        index=0 if model_ready else 1,
        help="The ensemble combines all three detectors and fuses their scores.",
    )
    if not model_ready:
        st.warning("No fine-tuned model found; falling back to the baseline.")
    engines = {
        "Ensemble (rules + Presidio + model)": ("rules", "presidio", "model"),
        "Baseline only (rules + Presidio)": ("rules", "presidio"),
        "Rules only": ("rules",),
    }[engine_choice]

    mode = st.selectbox(
        "Redaction mode",
        ["mask", "partial", "pseudonymize"],
        format_func={
            "mask": "Marker — [NAME]",
            "partial": "Partial — keeps the last digits",
            "pseudonymize": "Pseudonym — [NAME:a3f9c2]",
        }.get,
    )

    st.subheader("Queue thresholds")
    tau_auto = st.slider("Automatic redaction ≥", 0.5, 1.0, 0.90, 0.01)
    tau_discard = st.slider("Discard below", 0.0, 0.9, 0.50, 0.01)
    tau_high_risk = st.slider(
        "High risk: automatic ≥", 0.5, 1.0, 0.95, 0.01,
        help="Cards, accounts, identity documents and credentials.",
    )
    lang_choice = st.selectbox("Language", ["Auto-detect", "Spanish", "English"])

policy = ReviewPolicy(
    tau_auto=tau_auto, tau_discard=tau_discard, tau_high_risk=tau_high_risk
)

examples = load_examples()
tab_src, tab_upload = st.tabs(["Paste text / example", "Upload file"])
with tab_src:
    preset = st.selectbox("Preloaded example", ["(none)"] + list(examples))
    default = examples.get(preset, "")
    text = st.text_area("Document", value=default, height=230, placeholder="Paste the document here...")
with tab_upload:
    uploaded = st.file_uploader(".txt or .md file", type=["txt", "md"])
    if uploaded is not None:
        text = uploaded.read().decode("utf-8", errors="replace")
        st.success(f"{uploaded.name} — {len(text)} characters")

if not text.strip():
    st.info("Paste a document, upload a file, or pick an example to start.")
    st.stop()

lang = {"Spanish": "es", "English": "en"}.get(lang_choice) or detect_language(text)
pipeline = load_pipeline(engines)
pipeline.policy = policy

with st.spinner("Analysing..."):
    result = pipeline.process(text, lang=lang, mode=mode, enqueue=True)

cols = st.columns(5)
cols[0].metric("Entities", result.stats["n_entities"])
cols[1].metric("Auto-redacted", result.stats["n_auto_redacted"])
cols[2].metric("In review", result.stats["n_review"])
cols[3].metric("Discarded", result.stats["n_discarded"])
cols[4].metric("Verified leaks", len(result.leaks), delta=None)

if result.leaks:
    st.error(f"{len(result.leaks)} values were left unredacted: {result.leaks}")
else:
    st.success("Verification passed: no detected value survives in the redacted text.")

t1, t2, t3 = st.tabs(["Original vs redacted", "Detected entities", "Review queue"])

with t1:
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Original** · detected language: `{result.language}`")
        st.markdown(highlight(text, result.entities, result.decisions), unsafe_allow_html=True)
    with right:
        st.markdown(f"**Redacted** · mode `{mode}`")
        st.markdown(plain_block(result.redacted_text), unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download redacted text", result.redacted_text,
        file_name=f"{result.document_id}.redacted.txt", mime="text/plain",
    )
    d2.download_button(
        "Download audit JSON",
        json.dumps(result.audit_log(), ensure_ascii=False, indent=2, default=str),
        file_name=f"{result.document_id}.audit.json", mime="application/json",
    )

with t2:
    if result.entities:
        rows = []
        for ent in result.entities:
            decision = result.decisions.get(f"{ent.start}:{ent.end}:{ent.type}", "")
            rows.append(
                {
                    "Type": ent.type,
                    "Value": ent.text,
                    "Score": round(ent.score, 3),
                    "Decision": DECISION_STYLE.get(decision, (decision, ""))[0],
                    "Detectors": ", ".join(ent.metadata.get("detectors", [])) or ent.source,
                    "Reason": "; ".join(ent.metadata.get("decision_reasons", [])),
                    "Start": ent.start,
                    "End": ent.end,
                }
            )
        df = pd.DataFrame(rows).sort_values("Score")
        # Explicit widths: left to auto-size, the columns collapsed to a sliver.
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                "Type": st.column_config.TextColumn("Type", width="small"),
                "Value": st.column_config.TextColumn("Value", width="medium"),
                "Score": st.column_config.ProgressColumn(
                    "Score", min_value=0.0, max_value=1.0, format="%.3f", width="small"
                ),
                "Decision": st.column_config.TextColumn("Decision", width="small"),
                "Detectors": st.column_config.TextColumn("Detectors", width="small"),
                "Reason": st.column_config.TextColumn("Reason", width="large"),
                "Start": st.column_config.NumberColumn("Start", width="small"),
                "End": st.column_config.NumberColumn("End", width="small"),
            },
        )
        st.download_button(
            "Download entities (CSV)", df.to_csv(index=False),
            file_name=f"{result.document_id}.entities.csv", mime="text/csv",
        )
    else:
        st.info("No PII detected in this document.")

with t3:
    st.markdown(
        "Entities the pipeline is not confident enough to redact unsupervised. "
        "**They are redacted anyway**: a reviewer can put a false positive back, but a "
        "leak that has already shipped cannot be recalled."
    )
    if not result.review_items:
        st.success("No entity in this document requires review.")
    for i, item in enumerate(result.review_items):
        with st.container(border=True):
            head, act = st.columns([4, 1])
            head.markdown(
                f"**{item.entity_type}** · score `{item.score:.3f}`  \n"
                f"`{item.text}`  \n"
                f"<span style='color:#666;font-size:0.82em'>{html.escape(item.context)}</span>",
                unsafe_allow_html=True,
            )
            head.caption("Reason: " + "; ".join(item.reasons))
            if act.button("Confirm PII", key=f"ok-{i}", width="stretch"):
                st.session_state[f"verdict-{i}"] = ReviewStatus.APPROVED.value
            if act.button("False positive", key=f"no-{i}", width="stretch"):
                st.session_state[f"verdict-{i}"] = ReviewStatus.REJECTED.value
            verdict = st.session_state.get(f"verdict-{i}")
            if verdict:
                st.caption(f"Verdict recorded: **{verdict}**")

with st.expander("Processing statistics"):
    st.json(result.stats)
