"""
MedQuery: Interactive Command-Line Interface (CLI)
Enables conversational symptom input and adaptive differential diagnosis in the console.
"""

from pathlib import Path
from src.adaptive_questioner import AdaptiveQuestioner
from src.data_processor import DataProcessor
from src.diagnostic_engine import DiagnosticEngine
from src.explainability import ExplainabilityEngine
from src.knowledge_base import MedicalKnowledgeBase
from src.nlp_parser import SymptomNLPParser


def main():
    print("=" * 72)
    print("   MedQuery: Interactive Medical Differential Diagnosis Assistant")
    print("   Adaptive Evidence-Guided Clinical Decision Support Prototype")
    print("=" * 72)

    # Load artifacts
    processor = DataProcessor(data_dir="data")
    try:
        metadata = processor.load_metadata()
    except FileNotFoundError:
        print("\nMetadata not found. Running training and preprocessing pipeline first...\n")
        from train import run_pipeline
        run_pipeline()
        metadata = processor.load_metadata()

    symptom_columns = metadata["symptom_columns"]
    symptom_display_map = metadata["symptom_display_map"]
    disease_profile = metadata["disease_profile"]

    model_path = Path("models/medquery_model.pkl")
    if not model_path.exists():
        from train import run_pipeline
        run_pipeline()

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

    print("\nPlease describe your symptoms in your own words (or type comma-separated symptoms):")
    print("Example: 'I have severe headache with chills, shivering, and high fever since yesterday.'")
    user_input = input("\nSymptoms > ").strip()

    if not user_input:
        print("No input provided. Exiting.")
        return

    parsed = nlp.parse_text(user_input)
    present_symptoms = list(parsed["confirmed"])
    denied_symptoms = list(parsed["negated"])

    print("\n" + "-" * 72)
    print("Extracted Clinical Evidence:")
    if present_symptoms:
        print(f"  [+] Confirmed Symptoms: {', '.join([symptom_display_map.get(s, s) for s in present_symptoms])}")
    else:
        print("  [!] No recognized clinical symptoms found in the statement.")
        print("      Please try again with common symptom terms like fever, headache, cough, etc.")
        return

    if denied_symptoms:
        print(f"  [-] Denied Symptoms   : {', '.join([symptom_display_map.get(s, s) for s in denied_symptoms])}")

    asked_symptoms = set(present_symptoms + denied_symptoms)
    question_count = 0

    while True:
        # 1. Update differential
        diff_res = engine.predict_differential(present_symptoms, denied_symptoms, top_k=5)
        top_candidates = diff_res["differential"]

        print("\n" + "=" * 72)
        print(f"Current Dynamic Differential Diagnosis (Turn {question_count}):")
        for rank, item in enumerate(top_candidates[:3], 1):
            bar = "#" * int(item["probability"] * 25)
            print(f"  {rank}. {item['disease']:<38} {item['percentage']:>6} [{bar:<25}]")
        print(f"  Diagnostic Confidence: {diff_res['confidence_band']} | Entropy: {diff_res['entropy']} bits")

        # 2. Ask next adaptive question
        next_q = questioner.select_next_question(diff_res, asked_symptoms, question_count)
        if not next_q:
            print("\n>> Diagnostic threshold reached or maximum questions completed.")
            break

        print("\n" + "-" * 72)
        print(f"Follow-up Question {next_q['question_index']} of {next_q['total_questions_allowed']}:")
        print(f">> {next_q['question_text']}")
        try:
            ans = input("Answer (y = Yes, n = No, s = Skip / Not sure, q = Quit): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n>> Input completed or interrupted. Generating final report...\n")
            break

        sym_key = next_q["symptom_key"]
        asked_symptoms.add(sym_key)
        question_count += 1

        if ans in ["y", "yes"]:
            present_symptoms.append(sym_key)
        elif ans in ["n", "no"]:
            denied_symptoms.append(sym_key)
        elif ans in ["q", "quit"]:
            break
        else:
            # Skip / Not sure
            pass

    # Final detailed diagnostic report and explanation
    final_diff = engine.predict_differential(present_symptoms, denied_symptoms, top_k=5)
    explanation = explainer.generate_full_explanation(final_diff, present_symptoms, denied_symptoms)

    print("\n" + "=" * 72)
    print("                     FINAL MEDQUERY DIAGNOSTIC REPORT                   ")
    print("=" * 72)
    top_cand = explanation["top_candidate"]
    if top_cand:
        print(f"\nPrimary Diagnosis Candidate: {top_cand['disease'].upper()} ({top_cand['percentage']})")
        print(f"Urgency Level              : {top_cand['urgency']}")
        print(f"Recommended Specialist     : {top_cand['specialist']}")
        print(f"Clinical Summary           : {top_cand['description']}")

        print("\nSupporting Evidence (Observed Hallmark Symptoms):")
        if top_cand["supporting_symptoms"]:
            for s in top_cand["supporting_symptoms"]:
                print(f"  * {s['display_name']} ({s['importance']} - seen in {s['disease_frequency']} of cases)")
        else:
            print("  * Non-specific symptom presentation.")

        if top_cand["missing_hallmarks"]:
            print("\nCommon Symptoms of this Disease NOT Reported:")
            for s in top_cand["missing_hallmarks"][:4]:
                print(f"  o {s['display_name']} (typically present in {s['typical_frequency']})")

        print("\nRecommended Clinical Precautions:")
        for p in top_cand["precautions"]:
            print(f"  - {p}")

        if top_cand["warning_signs"]:
            print("\nRed Flag Warning Signs (Seek Immediate Medical Care if Present):")
            for w in top_cand["warning_signs"]:
                print(f"  ! {w}")

    print("\n" + "-" * 72)
    print("Disclaimer: MedQuery is an AI decision-support research prototype and does not")
    print("replace professional medical consultation, clinical examination, or laboratory testing.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
