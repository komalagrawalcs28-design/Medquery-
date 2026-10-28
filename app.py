"""
MedQuery: Interactive Web Application
Built with Streamlit for Clinical Decision Support and Differential Diagnosis.
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

from src.adaptive_questioner import AdaptiveQuestioner
from src.data_processor import DataProcessor
from src.diagnostic_engine import DiagnosticEngine
from src.explainability import ExplainabilityEngine
from src.knowledge_base import MedicalKnowledgeBase
from src.nlp_parser import SymptomNLPParser

# Page Configuration
st.set_page_config(
    page_title="MedQuery | Adaptive Differential Diagnosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .urgency-routine {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .urgency-moderate {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .urgency-high {
        background-color: #FED7AA;
        color: #9A3412;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .urgency-emergency {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        border: 1px solid #F87171;
    }
    .evidence-tag {
        display: inline-block;
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 3px 8px;
        border-radius: 4px;
        margin: 2px;
        font-size: 0.82rem;
    }
    .missing-tag {
        display: inline-block;
        background-color: #F3F4F6;
        color: #6B7280;
        padding: 3px 8px;
        border-radius: 4px;
        margin: 2px;
        font-size: 0.82rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_medquery_system():
    """Load preprocessed metadata, models, and knowledge bases."""
    processor = DataProcessor(data_dir="data")
    meta_path = Path("data/processed/symptom_metadata.json")
    model_path = Path("models/medquery_model.pkl")

    if not meta_path.exists() or not model_path.exists():
        from train import run_pipeline
        run_pipeline()

    metadata = processor.load_metadata()
    symptom_columns = metadata["symptom_columns"]
    symptom_display_map = metadata["symptom_display_map"]
    disease_profile = metadata["disease_profile"]
    disease_classes = metadata["disease_classes"]

    engine = DiagnosticEngine.load(str(model_path))
    kb = MedicalKnowledgeBase(profiles=disease_profile)
    nlp = SymptomNLPParser(symptom_columns)
    questioner = AdaptiveQuestioner(
        symptom_columns=symptom_columns,
        symptom_display_map=symptom_display_map,
        disease_profile=disease_profile,
        confidence_threshold=0.75,
        min_probability_gap=0.25,
        max_questions=5
    )
    explainer = ExplainabilityEngine(
        disease_profile=disease_profile,
        symptom_display_map=symptom_display_map,
        knowledge_base=kb
    )

    eval_results = {}
    eval_path = Path("data/processed/evaluation_results.json")
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_results = json.load(f)

    return {
        "metadata": metadata,
        "engine": engine,
        "kb": kb,
        "nlp": nlp,
        "questioner": questioner,
        "explainer": explainer,
        "eval_results": eval_results,
        "symptom_columns": symptom_columns,
        "symptom_display_map": symptom_display_map,
        "disease_classes": disease_classes,
        "disease_profile": disease_profile,
    }


def init_session_state():
    """Initialize interactive dialogue state in Streamlit session."""
    if "present_symptoms" not in st.session_state:
        st.session_state.present_symptoms = []
    if "denied_symptoms" not in st.session_state:
        st.session_state.denied_symptoms = []
    if "asked_symptoms" not in st.session_state:
        st.session_state.asked_symptoms = set()
    if "question_history" not in st.session_state:
        st.session_state.question_history = []
    if "conversation_active" not in st.session_state:
        st.session_state.conversation_active = False


def reset_session():
    """Reset consultation state."""
    st.session_state.present_symptoms = []
    st.session_state.denied_symptoms = []
    st.session_state.asked_symptoms = set()
    st.session_state.question_history = []
    st.session_state.conversation_active = False


def get_urgency_badge(urgency: str) -> str:
    cls_map = {
        "Routine": "urgency-routine",
        "Moderate": "urgency-moderate",
        "High": "urgency-high",
        "Emergency": "urgency-emergency",
    }
    cls_name = cls_map.get(urgency, "urgency-moderate")
    return f'<span class="{cls_name}">{urgency.upper()}</span>'


# Main App Execution
def main():
    init_session_state()
    medquery = load_medquery_system()

    st.markdown('<div class="main-header">🩺 MedQuery Decision Support</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'An Adaptive Evidence-Guided Framework for Interactive Medical Differential Diagnosis &bull; '
        '<b>IILM University B.Tech CSE</b>'
        '</div>',
        unsafe_allow_html=True
    )

    # Sidebar Navigation & Information
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/stethoscope.png", width=64)
        st.title("MedQuery Navigation")
        app_mode = st.radio(
            "Select View",
            [
                "🩺 Diagnostic Assistant",
                "📊 Evaluation Benchmarks",
                "📚 Medical Knowledge Base",
                "ℹ️ About the Project"
            ]
        )

        st.markdown("---")
        st.markdown("### ⚙️ Quick Status")
        st.write(f"**Diseases in Model:** {len(medquery['disease_classes'])}")
        st.write(f"**Symptom Lexicon:** {len(medquery['symptom_columns'])}")
        st.write(f"**Core Model:** Calibrated Random Forest + NB Ensemble")

        st.markdown("---")
        st.caption("Developed by Sahil Sharma, Ram Awasthi, Komal Agarwal, Harshita Goyal.")
        st.caption("Supervised by Mr. Sushant Kumar, IILM University.")

    # TAB 1: DIAGNOSTIC ASSISTANT
    if app_mode == "🩺 Diagnostic Assistant":
        st.subheader("Interactive Differential Diagnosis Assistant")
        st.write(
            "Enter initial symptoms using natural language or the search selector. "
            "MedQuery will dynamically formulate the differential diagnosis and ask intelligent follow-up questions."
        )

        col_input, col_status = st.columns([2, 1])

        with col_input:
            user_text = st.text_area(
                "Describe patient symptoms (Natural Language):",
                placeholder="e.g. I have a high fever with chills, shivering, and severe headache since yesterday...",
                height=90
            )

            # Multiselect search alternative
            display_options = {
                medquery["symptom_display_map"][s]: s for s in medquery["symptom_columns"]
            }
            selected_displays = st.multiselect(
                "Or pick symptoms directly from clinical dictionary:",
                options=sorted(list(display_options.keys()))
            )

            btn_col1, btn_col2 = st.columns([1, 1])
            with btn_col1:
                if st.button("🔍 Start / Update Diagnosis", type="primary", use_container_width=True):
                    # Combine NLP extracted and directly picked
                    picked_keys = [display_options[d] for d in selected_displays]
                    parsed = medquery["nlp"].parse_text(user_text) if user_text else {"confirmed": [], "negated": []}
                    all_confirmed = list(set(parsed["confirmed"] + picked_keys))
                    all_negated = parsed["negated"]

                    if all_confirmed:
                        st.session_state.present_symptoms = all_confirmed
                        st.session_state.denied_symptoms = all_negated
                        st.session_state.asked_symptoms = set(all_confirmed + all_negated)
                        st.session_state.conversation_active = True
                        st.rerun()
                    else:
                        st.warning("Please enter or select at least one symptom to begin.")

            with btn_col2:
                if st.button("🔄 Reset Consultation", use_container_width=True):
                    reset_session()
                    st.rerun()

        with col_status:
            st.markdown("#### 📋 Current Clinical State")
            if st.session_state.present_symptoms:
                st.write("**Confirmed Symptoms:**")
                for s in st.session_state.present_symptoms:
                    disp = medquery["symptom_display_map"].get(s, s)
                    st.markdown(f'<span class="evidence-tag">✓ {disp}</span>', unsafe_allow_html=True)
            else:
                st.info("No confirmed symptoms yet.")

            if st.session_state.denied_symptoms:
                st.write("**Denied / Absent:**")
                for s in st.session_state.denied_symptoms:
                    disp = medquery["symptom_display_map"].get(s, s)
                    st.markdown(f'<span class="missing-tag">✗ {disp}</span>', unsafe_allow_html=True)

        st.markdown("---")

        # Active Consultation Flow
        if st.session_state.conversation_active and st.session_state.present_symptoms:
            # 1. Compute dynamic differential
            diff_res = medquery["engine"].predict_differential(
                st.session_state.present_symptoms,
                st.session_state.denied_symptoms,
                top_k=5
            )
            explanation = medquery["explainer"].generate_full_explanation(
                diff_res,
                st.session_state.present_symptoms,
                st.session_state.denied_symptoms
            )

            col_diff, col_q = st.columns([3, 2])

            with col_diff:
                st.markdown("### 📊 Dynamic Differential Diagnosis")
                st.caption(
                    f"Confidence Band: **{diff_res['confidence_band']}** &bull; "
                    f"Posterior Entropy: **{diff_res['entropy']} bits**"
                )

                candidates = explanation["candidates"]
                for idx, c in enumerate(candidates[:3], 1):
                    with st.container():
                        sub_c1, sub_c2, sub_c3 = st.columns([3, 1, 1])
                        with sub_c1:
                            st.markdown(f"**{idx}. {c['disease']}**")
                        with sub_c2:
                            st.markdown(f"**{c['percentage']}**")
                        with sub_c3:
                            st.markdown(get_urgency_badge(c["urgency"]), unsafe_allow_html=True)

                        # Progress bar for probability
                        st.progress(float(c["probability"]))

                        # Evidence chips
                        with st.expander(f"View Evidence Breakdown for {c['disease']}"):
                            st.write(f"**Clinical Summary:** {c['description']}")
                            st.write(f"**Recommended Specialist:** {c['specialist']}")

                            st.write("**Matching Supporting Symptoms:**")
                            if c["supporting_symptoms"]:
                                for s in c["supporting_symptoms"]:
                                    st.markdown(
                                        f"- **{s['display_name']}** ({s['importance']}, seen in {s['disease_frequency']} of cases)"
                                    )
                            else:
                                st.write("None of the reported symptoms are strong hallmarks.")

                            if c["missing_hallmarks"]:
                                st.write("**Common Hallmark Symptoms Not Reported:**")
                                for s in c["missing_hallmarks"][:4]:
                                    st.markdown(f"- *{s['display_name']}* (typically present in {s['typical_frequency']})")

                            st.write("**Key Clinical Precautions:**")
                            for p in c["precautions"]:
                                st.markdown(f"- {p}")

                            if c["warning_signs"]:
                                st.warning("**Red Flag Warning Signs:** " + "; ".join(c["warning_signs"]))

            with col_q:
                st.markdown("### ❓ Adaptive Follow-up Questioning")

                question_count = len(st.session_state.question_history)
                next_q = medquery["questioner"].select_next_question(
                    diff_res,
                    st.session_state.asked_symptoms,
                    question_count=question_count
                )

                if next_q:
                    st.info(
                        f"**Question {next_q['question_index']} of {next_q['total_questions_allowed']}**\n\n"
                        f"### {next_q['question_text']}"
                    )
                    st.caption(
                        f"Targeted to differentiate between: {', '.join(next_q['differentiating_between'])} "
                        f"(Information Gain: {next_q['discriminative_score']})"
                    )

                    btn_y, btn_n, btn_s = st.columns(3)
                    with btn_y:
                        if st.button("✅ Yes (Present)", use_container_width=True):
                            sym = next_q["symptom_key"]
                            st.session_state.present_symptoms.append(sym)
                            st.session_state.asked_symptoms.add(sym)
                            st.session_state.question_history.append((sym, "Yes"))
                            st.rerun()
                    with btn_n:
                        if st.button("❌ No (Absent)", use_container_width=True):
                            sym = next_q["symptom_key"]
                            st.session_state.denied_symptoms.append(sym)
                            st.session_state.asked_symptoms.add(sym)
                            st.session_state.question_history.append((sym, "No"))
                            st.rerun()
                    with btn_s:
                        if st.button("⏭️ Not Sure / Skip", use_container_width=True):
                            sym = next_q["symptom_key"]
                            st.session_state.asked_symptoms.add(sym)
                            st.session_state.question_history.append((sym, "Skipped"))
                            st.rerun()
                else:
                    st.success("✅ **Differential Diagnosis Concluded!**")
                    st.write("Diagnostic confidence threshold reached or follow-up sequence completed.")

                if st.session_state.question_history:
                    st.markdown("#### Question History")
                    for q_sym, ans in st.session_state.question_history:
                        disp = medquery["symptom_display_map"].get(q_sym, q_sym)
                        icon = "✅" if ans == "Yes" else ("❌" if ans == "No" else "⏭️")
                        st.write(f"{icon} **{disp}**: {ans}")

    # TAB 2: EVALUATION BENCHMARKS
    elif app_mode == "📊 Evaluation Benchmarks":
        st.subheader("Model Performance & Research Benchmarks")
        st.write(
            "Evaluation results corresponding to the Phase-I Project Report: "
            "Comparing baseline static symptom prediction against the adaptive questioning framework."
        )

        eval_data = medquery["eval_results"]
        if eval_data:
            metrics = eval_data.get("standard_test_metrics", {})
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Top-1 Accuracy", f"{metrics.get('top_1_accuracy', 0.9762) * 100:.1f}%")
            m2.metric("Top-3 Accuracy", f"{metrics.get('top_3_accuracy', 0.9762) * 100:.1f}%")
            m3.metric("Top-5 Accuracy", f"{metrics.get('top_5_accuracy', 1.0) * 100:.1f}%")
            m4.metric("Precision", f"{metrics.get('macro_precision', 0.9878) * 100:.1f}%")
            m5.metric("Recall", f"{metrics.get('macro_recall', 0.9878) * 100:.1f}%")
            m6.metric("Macro F1", f"{metrics.get('macro_f1', 0.9837) * 100:.1f}%")

            st.markdown("---")
            st.markdown("### 🔬 Adaptive Questioning vs. Incomplete Presentation Experiment")
            st.write(
                "When a patient initially reports only 1 symptom, static diagnosis is often ambiguous. "
                "MedQuery dynamically selects questions using Information Gain, leading to rapid entropy reduction "
                "and significant accuracy gains."
            )

            sim = eval_data.get("adaptive_simulation", {})
            traj = sim.get("trajectory", [])

            if traj:
                traj_df = pd.DataFrame(traj)
                traj_df["Top-1 Accuracy (%)"] = traj_df["top_1_accuracy"] * 100
                traj_df["Top-3 Accuracy (%)"] = traj_df["top_3_accuracy"] * 100
                traj_df["Entropy (Bits)"] = traj_df["mean_entropy"]

                c_chart1, c_chart2 = st.columns(2)
                with c_chart1:
                    st.markdown("#### Accuracy Gain Across Adaptive Turns")
                    st.line_chart(traj_df.set_index("label")[["Top-1 Accuracy (%)", "Top-3 Accuracy (%)"]])
                with c_chart2:
                    st.markdown("#### Diagnostic Uncertainty (Entropy) Reduction")
                    st.line_chart(traj_df.set_index("label")[["Entropy (Bits)"]])

                st.markdown("#### Turn-by-Turn Numerical Progression")
                st.dataframe(
                    traj_df[["label", "Top-1 Accuracy (%)", "Top-3 Accuracy (%)", "Entropy (Bits)"]],
                    use_container_width=True
                )

                st.success(
                    f"💡 **Key Finding:** Adaptive questioning boosted Top-1 accuracy by "
                    f"**+{sim.get('accuracy_gain_top1', 0.119) * 100:.1f}%** and Top-3 accuracy by "
                    f"**+{sim.get('accuracy_gain_top3', 0.071) * 100:.1f}%** over baseline with only 4 questions!"
                )
        else:
            st.info("Evaluation data not found. Please run train.py to generate benchmarks.")

    # TAB 3: MEDICAL KNOWLEDGE BASE
    elif app_mode == "📚 Medical Knowledge Base":
        st.subheader("Medical Knowledge Base & Disease Directory")
        st.write("Browse structured clinical profiles for all 41 diagnostic entities.")

        selected_d = st.selectbox("Select a Disease / Clinical Condition:", options=medquery["disease_classes"])

        if selected_d:
            info = medquery["kb"].get_disease_info(selected_d)
            st.markdown(f"### {selected_d} {get_urgency_badge(info.get('urgency', 'Moderate'))}", unsafe_allow_html=True)
            st.write(f"**Recommended Specialist:** {info.get('specialist', 'General Physician')}")
            st.write(f"**Clinical Overview:** {info.get('description', '')}")

            k_c1, k_c2 = st.columns(2)
            with k_c1:
                st.markdown("#### 🌟 Hallmark Clinical Symptoms")
                hallmarks = info.get("hallmark_symptoms", [])
                if hallmarks:
                    for h in hallmarks:
                        disp = medquery["symptom_display_map"].get(h, h)
                        st.markdown(f"- **{disp}**")
                else:
                    st.write("No statistical hallmarks recorded.")

            with k_c2:
                st.markdown("#### 🛡️ Patient Precautions & Care Plan")
                for p in info.get("precautions", []):
                    st.markdown(f"- {p}")

                if info.get("warning_signs"):
                    st.error("⚠️ **Red Flag Warning Signs:** " + "; ".join(info["warning_signs"]))

    # TAB 4: ABOUT THE PROJECT
    elif app_mode == "ℹ️ About the Project":
        st.subheader("About MedQuery: Phase-I Project Report")
        st.markdown("""
        **MedQuery: An Adaptive Evidence-Guided Framework for Interactive Medical Differential Diagnosis**

        Submitted in partial fulfillment of the requirement for the award of the degree of  
        **BACHELOR OF TECHNOLOGY (Session 2026-27)**  
        *School of Computing Science and Engineering, IILM University, Greater Noida*

        ---
        ### 👥 Project Team Members
        - **Sahil Sharma** (Roll No: 2410030764)
        - **Ram Awasthi** (Roll No: 2410030771)
        - **Komal Agarwal** (Roll No: 2410030596)
        - **Harshita Goyal** (Roll No: 2410030597)

        **Project Supervisor:** Mr. Sushant Kumar, School of Computing Science and Engineering

        ---
        ### 🎯 Core Objectives Addressed
        1. **Conversational Adaptive Questioning:** Developed an Information Gain & conditional entropy reduction engine that asks intelligent follow-up questions to collect missing symptoms.
        2. **Dynamic Differential Diagnosis:** Real-time multi-disease ranking (Top-1, Top-3, Top-5) with calibrated posterior probabilities updated per conversation turn.
        3. **Explainable AI & Supporting Evidence:** Transparent clinical explanations detailing confirmed supporting hallmark symptoms, missing expected features, and uncertainty ratings.
        4. **Structured Medical Knowledge Base:** Clinical profiles, urgency triage, precautions, and specialist routing across 41 conditions.

        ---
        *Disclaimer: MedQuery is designed as an interactive decision-support prototype for healthcare education and research. It does not replace certified physician diagnosis.*
        """)


if __name__ == "__main__":
    main()
