"""HealthBandhu V: Streamlit web application.

Start it from the project root (the folder that contains this file):

    streamlit run App.py

The trained model files must be in the models/ folder. They are produced by
notebooks/HealthBandhu_Training.ipynb.
"""

from __future__ import annotations

import html
import json
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from healthbandhu import config
from healthbandhu.models import ArtifactsMissingError, DiagnosisResult, HealthBandhuPredictor, load_bundle
from healthbandhu.report import build_pdf_report
from healthbandhu.symptom_groups import display_name, group_symptoms

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Visual language
# ---------------------------------------------------------------------------
INK = "#1F2544"
INDIGO = "#2F3E8F"
MUTED = "#5B6178"
RULE = "#D9DDEA"
AGAINST = "#A3A9BC"

TRIAGE_STYLE = {
    "CRITICAL": ("#B42318", "#FEF3F2"),
    "HIGH RISK": ("#C2410C", "#FFF4ED"),
    "MODERATE RISK": ("#A16207", "#FEFBE8"),
    "LOW RISK": ("#15803D", "#F0FDF4"),
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Hind+Siliguri:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

.hb-triage, .hb-lead, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
  font-family: 'Hind Siliguri', 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}}
.hb-triage {{
  border-left: 10px solid var(--hb-c);
  background: var(--hb-bg);
  padding: 1.1rem 1.4rem 1.2rem;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
  margin: .25rem 0 1.25rem;
}}
.hb-triage .hb-level {{
  font-size: 2.1rem;
  font-weight: 700;
  line-height: 1.05;
  color: var(--hb-c);
  letter-spacing: -0.02em;
}}
.hb-triage .hb-action {{
  font-size: 1.08rem;
  color: {INK};
  margin-top: .45rem;
  max-width: 70ch;
  line-height: 1.5;
}}
.hb-triage .hb-why {{
  font-size: .92rem;
  color: {MUTED};
  margin-top: .35rem;
}}
.hb-lead {{
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  column-gap: 1.75rem;
  row-gap: .25rem;
  padding-bottom: .85rem;
  border-bottom: 1px solid {RULE};
  margin-bottom: .5rem;
}}
.hb-lead .hb-name {{
  font-size: 1.75rem;
  font-weight: 700;
  color: {INK};
  letter-spacing: -0.01em;
}}
.hb-lead .hb-prob {{
  font-size: 1.75rem;
  font-weight: 700;
  color: {INDIGO};
}}
.hb-lead .hb-meta {{
  font-size: .98rem;
  color: {MUTED};
}}
</style>
"""

EXAMPLES = {
    "Chest pain with breathlessness": ["sharp chest pain", "shortness of breath", "dizziness", "sweating"],
    "Cough, fever and sore throat": ["cough", "fever", "sore throat", "nasal congestion", "headache"],
    "Painful, frequent urination": ["painful urination", "frequent urination", "suprapubic pain", "blood in urine"],
    "Itchy skin rash": ["skin rash", "itching of skin", "abnormal appearing skin", "skin lesion"],
    "Stomach pain with vomiting": ["sharp abdominal pain", "vomiting", "nausea", "diarrhea", "fever"],
}
EXAMPLE_PROMPT = "Load an example symptom set"
PAGES = ("Symptom check", "Model performance", "About")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading HealthBandhu models")
def get_predictor(models_dir: str) -> HealthBandhuPredictor:
    return HealthBandhuPredictor(load_bundle(models_dir))


def pct(p) -> str:
    return "-" if p is None else f"{p * 100:.1f}%"


# ---------------------------------------------------------------------------
# Session callbacks
# ---------------------------------------------------------------------------
def load_example(available: set) -> None:
    choice = st.session_state.get("example_choice")
    if choice in EXAMPLES:
        st.session_state["symptoms"] = [s for s in EXAMPLES[choice] if s in available]
    st.session_state["example_choice"] = EXAMPLE_PROMPT


def clear_form() -> None:
    st.session_state["symptoms"] = []
    st.session_state.pop("result", None)
    st.session_state.pop("pdf", None)


def add_from_group(widget_key: str) -> None:
    current = list(st.session_state.get("symptoms", []))
    for s in st.session_state.get(widget_key, []):
        if s not in current:
            current.append(s)
    st.session_state["symptoms"] = current
    st.session_state[widget_key] = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(predictor):
    with st.sidebar:
        st.markdown(f"## {config.APP_TITLE}")
        st.caption(config.APP_TAGLINE)
        page = st.radio("Section", PAGES, label_visibility="collapsed")
        st.divider()
        if predictor is not None:
            b = predictor.bundle
            st.markdown("**Loaded models**")
            lines = []
            if predictor.has_dnn:
                lines.append(f"Deep neural network, weight {b.dnn_weight:.2f}")
            if predictor.has_nb:
                lines.append(f"Bernoulli Naive Bayes, weight {1 - b.dnn_weight:.2f}")
            lines.append("Rule-based cross-check" if predictor.has_rules else "Rule-based cross-check unavailable")
            st.markdown("\n".join(f"- {line}" for line in lines))
            st.caption(f"{len(predictor.symptoms)} symptoms, {len(predictor.diseases)} conditions")
            for note in b.warnings:
                if "trained with" in note:
                    st.warning(note)
        st.divider()
        st.caption(config.DISCLAIMER)
    return page


# ---------------------------------------------------------------------------
# Setup screen (no models yet)
# ---------------------------------------------------------------------------
def render_setup(err: ArtifactsMissingError) -> None:
    st.title(config.APP_TITLE)
    st.error(f"No trained models found in `{err.models_dir}`.")
    st.markdown(
        """
The app needs the files created by the training notebook.

1. Open `notebooks/HealthBandhu_Training.ipynb` in Google Colab or Jupyter.
2. Run all cells. Training saves the model files into the `models` folder.
3. Copy the `models` folder into this project (skip if you trained here), then restart with `streamlit run App.py`.
"""
    )
    st.markdown("**Missing files**")
    st.code("\n".join(err.missing), language="text")


# ---------------------------------------------------------------------------
# Symptom check
# ---------------------------------------------------------------------------
def render_symptom_check(predictor: HealthBandhuPredictor) -> None:
    st.title("Symptom check")
    st.markdown(
        "Select every symptom the patient has. HealthBandhu screens for red-flag symptoms first, "
        "then ranks possible conditions and shows which symptoms drove the result."
    )

    available = set(predictor.symptoms)
    st.session_state.setdefault("symptoms", [])
    st.session_state.setdefault("example_choice", EXAMPLE_PROMPT)

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Patient")
        name = st.text_input("Name (optional)", key="patient_name", max_chars=80)
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", min_value=0, max_value=120, value=None, step=1,
                              placeholder="Years", key="patient_age")
        sex = c2.selectbox("Sex", ["Not specified", "Female", "Male", "Other"], key="patient_sex")

    with right:
        st.subheader("Symptoms")
        st.multiselect(
            "Symptoms",
            sorted(predictor.symptoms),
            key="symptoms",
            format_func=display_name,
            placeholder="Type to search, for example fever, cough or chest pain",
            label_visibility="collapsed",
        )
        c1, c2 = st.columns([3, 1])
        examples = [k for k, v in EXAMPLES.items() if any(s in available for s in v)]
        c1.selectbox("Example", [EXAMPLE_PROMPT] + examples, key="example_choice",
                     on_change=load_example, args=(available,), label_visibility="collapsed")
        c2.button("Clear all", on_click=clear_form, width="stretch")

        with st.expander("Browse symptoms by body system"):
            groups = group_symptoms(predictor.symptoms)
            group = st.selectbox("Body system", list(groups),
                                 format_func=lambda g: f"{g} ({len(groups[g])})")
            widget_key = "add_" + "".join(ch if ch.isalnum() else "_" for ch in group.lower())
            st.multiselect(f"Add symptoms from {group.lower()}", groups[group], key=widget_key,
                           format_func=display_name, on_change=add_from_group, args=(widget_key,),
                           placeholder="Choose symptoms to add")

    selected = list(st.session_state.get("symptoms", []))
    count = len(selected)
    run = st.button(
        f"Analyse {count} symptom{'s' if count != 1 else ''}" if count else "Analyse symptoms",
        type="primary", disabled=count == 0, width="stretch",
    )
    if count == 0 and "result" not in st.session_state:
        st.caption("Choose at least one symptom to start. Results are more reliable with three or more.")

    if run:
        with st.spinner("Analysing symptoms"):
            st.session_state["result"] = predictor.diagnose(selected)
            st.session_state["patient"] = {
                "name": name.strip(),
                "age": "" if age is None else int(age),
                "sex": "" if sex == "Not specified" else sex,
            }
            st.session_state.pop("pdf", None)

    result = st.session_state.get("result")
    if result is not None:
        if sorted(result.symptoms) != sorted(selected):
            st.info("The symptom list has changed since this result. Select Analyse to update it.")
        render_result(result, predictor)


def render_result(result: DiagnosisResult, predictor: HealthBandhuPredictor) -> None:
    st.divider()
    em = result.emergency
    color, background = TRIAGE_STYLE[em.level]
    why = (
        "Red-flag symptoms: " + ", ".join(f"{html.escape(s)} (+{w})" for s, w in em.triggers)
        if em.triggers else "No red-flag symptoms among those reported"
    )
    contact = f" {html.escape(config.EMERGENCY_CONTACT_NOTE)}" if em.is_urgent else ""
    st.markdown(
        f"""<div class="hb-triage" style="--hb-c:{color};--hb-bg:{background}">
  <div class="hb-level">{html.escape(em.level.title())}</div>
  <div class="hb-action">{html.escape(em.action)}{contact}</div>
  <div class="hb-why">{why}. Emergency score {em.score}.</div>
</div>""",
        unsafe_allow_html=True,
    )

    top = result.top
    conf = result.confidence
    meta = f"{conf.level.title()} confidence"
    if conf.ambiguous:
        meta += f", only {conf.margin * 100:.1f} points ahead of the next condition"
    st.markdown(
        f"""<div class="hb-lead">
  <span class="hb-name">{html.escape(display_name(top.name))}</span>
  <span class="hb-prob">{pct(top.probability)}</span>
  <span class="hb-meta">{html.escape(meta)}</span>
</div>""",
        unsafe_allow_html=True,
    )
    if result.unknown_symptoms:
        st.warning("Not recognised and ignored: " + ", ".join(result.unknown_symptoms))

    tab_conditions, tab_why, tab_rules, tab_next = st.tabs(
        ["Possible conditions", "Why this result", "Rule-based cross-check", "Next steps and report"]
    )

    with tab_conditions:
        rows = []
        for c in result.candidates:
            rows.append({
                "Condition": display_name(c.name),
                "Probability": c.probability * 100,
                "Deep neural network": None if c.dnn_probability is None else c.dnn_probability * 100,
                "Naive Bayes": None if c.nb_probability is None else c.nb_probability * 100,
                "Rule score": c.rule_score,
                "Training cases": c.train_count,
            })
        df = pd.DataFrame(rows).dropna(axis=1, how="all")
        st.dataframe(
            df, hide_index=True, width="stretch",
            column_config={
                "Probability": st.column_config.ProgressColumn("Probability", format="%.1f%%", min_value=0,
                                                               max_value=100),
                "Deep neural network": st.column_config.NumberColumn(format="%.1f%%"),
                "Naive Bayes": st.column_config.NumberColumn(format="%.1f%%"),
                "Rule score": st.column_config.NumberColumn(format="%.2f",
                                                            help="Symptom-profile match from 0 to 1"),
                "Training cases": st.column_config.NumberColumn(format="%d"),
            },
        )
        if predictor.has_dnn and predictor.has_nb:
            st.caption(
                f"Probability = {result.dnn_weight:.2f} x deep neural network + "
                f"{1 - result.dnn_weight:.2f} x Naive Bayes. The weight was tuned on validation data."
            )
        if result.dnn_top and result.nb_top and result.dnn_top != result.nb_top:
            st.info(
                f"The models disagree: the deep neural network favours {display_name(result.dnn_top)}, "
                f"Naive Bayes favours {display_name(result.nb_top)}."
            )

    with tab_why:
        st.markdown(
            f"Each reported symptom was removed in turn to see how much the evidence for "
            f"**{display_name(top.name)}** weakens without it. Bars show each symptom's share of that "
            "evidence; a negative share means the symptom argues against this condition."
        )
        contrib = pd.DataFrame(
            [{"Symptom": display_name(c.symptom), "Influence": round(c.influence * 100, 1),
              "Probability change": round(c.impact * 100, 2),
              "Effect": "Supports" if c.influence >= 0 else "Argues against"}
             for c in result.contributions[:15]]
        )
        if contrib.empty or contrib["Influence"].abs().max() == 0:
            st.info("Removing any single symptom leaves this result unchanged, so no one symptom drives it.")
        else:
            chart = (
                alt.Chart(contrib)
                .mark_bar(cornerRadiusEnd=2)
                .encode(
                    x=alt.X("Influence:Q", title="Share of the evidence (%)"),
                    y=alt.Y("Symptom:N", sort=None, title=None),
                    color=alt.Color("Effect:N", scale=alt.Scale(domain=["Supports", "Argues against"],
                                                                range=[INDIGO, AGAINST]),
                                    legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["Symptom", alt.Tooltip("Influence:Q", format=".1f", title="Influence (%)"),
                             alt.Tooltip("Probability change:Q", format=".2f",
                                         title="Probability drop if removed (points)")],
                )
                .properties(height=max(120, 34 * len(contrib)))
            )
            st.altair_chart(chart, width="stretch")
        if result.unreported_typical:
            st.markdown(f"**Often seen with {display_name(top.name)} but not reported**")
            st.markdown("\n".join(
                f"- {display_name(s)}: present in {p * 100:.0f}% of training cases"
                for s, p in result.unreported_typical
            ))
            st.caption("If the patient has any of these, add them and analyse again.")

    with tab_rules:
        if not result.rule_matches:
            st.info("Disease profiles were not found, so the rule-based cross-check is unavailable.")
        else:
            st.markdown(
                "An independent check that compares the reported symptoms with each condition's key "
                "symptoms, without using the machine learning models."
            )
            rule_df = pd.DataFrame([
                {"Condition": display_name(m.disease), "Score": m.score,
                 "Key symptoms matched": f"{len(m.matched)} of {m.key_count}",
                 "Matched": ", ".join(m.matched) or "None",
                 "Key symptoms not reported": ", ".join(m.missing[:4]) or "None"}
                for m in result.rule_matches
            ])
            st.dataframe(rule_df, hide_index=True, width="stretch", column_config={
                "Score": st.column_config.ProgressColumn("Score", format="%.2f", min_value=0, max_value=1)})
            if result.rule_matches[0].disease == top.name:
                st.success("The rule-based check agrees with the model's top condition.")
            else:
                st.info(
                    f"The rule-based check ranks {display_name(result.rule_matches[0].disease)} first, "
                    "which differs from the models. Consider both."
                )

    with tab_next:
        st.markdown("\n".join(f"- {html.escape(r)}" for r in result.recommendations))
        patient = st.session_state.get("patient", {})
        if "pdf" not in st.session_state:
            st.session_state["pdf"] = build_pdf_report(result, patient)
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        c1, c2 = st.columns(2)
        c1.download_button("Download PDF report", data=st.session_state["pdf"],
                           file_name=f"healthbandhu_report_{stamp}.pdf", mime="application/pdf",
                           type="primary", width="stretch", on_click="ignore")
        payload = result.to_dict()
        payload["patient"] = patient
        c2.download_button("Download result as JSON", data=json.dumps(payload, indent=2),
                           file_name=f"healthbandhu_result_{stamp}.json", mime="application/json",
                           width="stretch", on_click="ignore")
    st.caption(config.DISCLAIMER)


# ---------------------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------------------
MODEL_ORDER = ["Rule-based", "Bernoulli Naive Bayes", "Extra Trees (baseline)", "Deep neural network", "Ensemble"]


def render_performance(predictor: HealthBandhuPredictor) -> None:
    st.title("Model performance")
    meta = predictor.bundle.metadata
    metrics = meta.get("metrics", {}).get("test", {})
    if not metrics:
        st.info(f"No performance data: `{config.METADATA_FILE}` is missing from the models folder.")
        return

    ds = meta.get("dataset", {})
    n_test = ds.get("test_rows")
    size = f"**{n_test:,}** " if isinstance(n_test, int) else ""
    st.markdown(f"All figures are measured on {size}held-out test patients that were never used for "
                "training or tuning.")

    rows = []
    for name in sorted(metrics, key=lambda n: MODEL_ORDER.index(n) if n in MODEL_ORDER else 99):
        m = metrics[name]
        rows.append({
            "Model": name,
            "Top-1 accuracy": m.get("top1", 0) * 100,
            "Top-3 accuracy": m.get("top3", 0) * 100,
            "Top-5 accuracy": m.get("top5", 0) * 100,
            "Macro F1": m.get("macro_f1"),
            "Weighted F1": m.get("weighted_f1"),
            "In app": "Yes" if m.get("deployed") else "No",
        })
    table = pd.DataFrame(rows)
    pc = dict(format="%.1f%%", min_value=0, max_value=100)
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "Top-1 accuracy": st.column_config.ProgressColumn("Top-1 accuracy", **pc),
        "Top-3 accuracy": st.column_config.ProgressColumn("Top-3 accuracy", **pc),
        "Top-5 accuracy": st.column_config.ProgressColumn("Top-5 accuracy", **pc),
        "Macro F1": st.column_config.NumberColumn(format="%.3f"),
        "Weighted F1": st.column_config.NumberColumn(format="%.3f"),
    })
    st.caption("Top-k accuracy: the correct condition appears among the k highest-ranked conditions.")

    long = table.melt(id_vars="Model", value_vars=["Top-1 accuracy", "Top-3 accuracy", "Top-5 accuracy"],
                      var_name="Metric", value_name="Accuracy")
    chart = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            y=alt.Y("Model:N", sort=list(table["Model"]), title=None),
            x=alt.X("Accuracy:Q", scale=alt.Scale(domain=[0, 100]), title="Accuracy (%)"),
            color=alt.Color("Metric:N", scale=alt.Scale(range=[INDIGO, "#6F7BC2", "#B9C0E4"]),
                            legend=alt.Legend(orient="bottom", title=None)),
            yOffset="Metric:N",
            tooltip=["Model", "Metric", alt.Tooltip("Accuracy:Q", format=".1f")],
        )
        .properties(height=70 * len(table))
    )
    st.altair_chart(chart, width="stretch")

    calib = meta.get("confidence_calibration", {})
    if calib:
        st.subheader("What the confidence levels mean")
        st.markdown("How often the top condition was correct on test patients, for each confidence level.")
        cal = pd.DataFrame([
            {"Confidence": level.title(), "Test patients": v.get("n", 0),
             "Share of patients": v.get("share", 0) * 100, "Top condition correct": v.get("accuracy", 0) * 100}
            for level, v in calib.items()
        ])
        st.dataframe(cal, hide_index=True, width="stretch", column_config={
            "Share of patients": st.column_config.NumberColumn(format="%.1f%%"),
            "Top condition correct": st.column_config.ProgressColumn("Top condition correct", **pc),
        })

    if ds:
        st.subheader("Dataset")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patient records used", f"{ds.get('rows_used', 0):,}")
        c2.metric("Symptoms", ds.get("symptoms_used", len(predictor.symptoms)))
        c3.metric("Conditions", ds.get("diseases_used", len(predictor.diseases)))
        c4.metric("Train / validation / test", f"{ds.get('train_rows', 0):,} / {ds.get('val_rows', 0):,} / "
                                               f"{ds.get('test_rows', 0):,}")
        notes = []
        if ds.get("merged_duplicate_columns"):
            notes.append("Merged duplicated symptom columns: " + ", ".join(
                f"{k} into {v}" for k, v in ds["merged_duplicate_columns"].items()))
        if ds.get("dropped_zero_activation_symptoms"):
            notes.append(f"Removed {len(ds['dropped_zero_activation_symptoms'])} symptoms that never occur in "
                         "the training data.")
        if ds.get("removed_rare_classes"):
            notes.append(f"Removed {len(ds['removed_rare_classes'])} conditions with fewer than "
                         f"{ds.get('min_samples_per_class', 2)} records.")
        if ds.get("exact_duplicate_rows") is not None:
            notes.append(f"{ds['exact_duplicate_rows']:,} records are exact duplicates of another record.")
        if notes:
            st.markdown("\n".join(f"- {n}" for n in notes))

    versions = meta.get("versions")
    if versions:
        with st.expander("Training environment"):
            st.json(versions)
            st.caption(f"Trained on {meta.get('created_at', 'unknown date')}.")


# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------
def render_about(predictor) -> None:
    st.title(f"About {config.APP_TITLE}")
    st.markdown(
        f"""
HealthBandhu is a clinical decision support prototype that ranks possible conditions from reported
symptoms. It was built by {config.AUTHOR} as an applied machine learning project.

**How a symptom check runs**

1. **Emergency screening.** Red-flag symptoms are scored before anything else and set the triage level.
2. **Deep neural network.** A multilayer network trained on the symptom dataset predicts a probability for every condition.
3. **Bernoulli Naive Bayes.** A classical probabilistic model gives an independent set of probabilities.
4. **Ensemble.** The two are blended with a weight tuned on validation data.
5. **Explanation.** Each symptom is removed in turn to measure its effect on the top result.
6. **Rule-based cross-check.** Symptoms are matched against each condition's key symptoms without machine learning.
7. **Confidence and recommendations.** The result is graded and turned into plain-language next steps.
8. **Report.** Everything is exported as a PDF or JSON file.
"""
    )
    st.warning(config.DISCLAIMER)
    if predictor is not None and predictor.bundle.warnings:
        with st.expander("Model loading notes"):
            st.markdown("\n".join(f"- {w}" for w in predictor.bundle.warnings))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    predictor = None
    load_error = None
    try:
        predictor = get_predictor(str(config.MODELS_DIR))
    except ArtifactsMissingError as err:
        load_error = err

    page = render_sidebar(predictor)

    if page == "About":
        render_about(predictor)
    elif load_error is not None:
        render_setup(load_error)
    elif page == "Model performance":
        render_performance(predictor)
    else:
        render_symptom_check(predictor)


main()
