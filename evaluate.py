"""
MedQuery: Standalone Evaluation Script
Loads trained models and outputs comprehensive diagnostic metrics & simulations.
"""

import json
from pathlib import Path
import pandas as pd
from src.adaptive_questioner import AdaptiveQuestioner
from src.data_processor import DataProcessor
from src.diagnostic_engine import DiagnosticEngine
from src.evaluation import MedQueryEvaluator


def run_evaluation():
    print("=" * 70)
    print(" MedQuery: Diagnostic Framework Evaluation Benchmark")
    print("=" * 70)

    processor = DataProcessor(data_dir="data")
    metadata = processor.load_metadata()

    symptom_columns = metadata["symptom_columns"]
    symptom_display_map = metadata["symptom_display_map"]
    disease_profile = metadata["disease_profile"]

    test_path = Path("data/processed/cleaned_test.csv")
    if not test_path.exists():
        print("Processed test data not found. Please run train.py first.")
        return

    test_df = pd.read_csv(test_path)

    model_path = Path("models/medquery_model.pkl")
    if not model_path.exists():
        print("Model file not found. Please run train.py first.")
        return

    engine = DiagnosticEngine.load(str(model_path))

    questioner = AdaptiveQuestioner(
        symptom_columns=symptom_columns,
        symptom_display_map=symptom_display_map,
        disease_profile=disease_profile,
        confidence_threshold=0.75,
        min_probability_gap=0.25,
        max_questions=5
    )

    evaluator = MedQueryEvaluator(engine, questioner, symptom_columns)

    print("\n--- Standard Performance on Held-Out Test Set (42 Cases) ---")
    results = evaluator.evaluate_test_set(test_df)
    print(f"Top-1 Accuracy : {results['top_1_accuracy'] * 100:.2f}%")
    print(f"Top-3 Accuracy : {results['top_3_accuracy'] * 100:.2f}%")
    print(f"Top-5 Accuracy : {results['top_5_accuracy'] * 100:.2f}%")
    print(f"Macro Precision: {results['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall   : {results['macro_recall'] * 100:.2f}%")
    print(f"Macro F1-Score : {results['macro_f1'] * 100:.2f}%")
    print(f"Weighted F1    : {results['weighted_f1'] * 100:.2f}%")

    print("\n--- Incomplete Presentation & Adaptive Follow-up Simulation ---")
    sim = evaluator.simulate_adaptive_questioning_experiment(
        test_df, num_initial_symptoms=1, max_followups=4
    )

    print(f"{'Step':<22} | {'Top-1 Acc':<12} | {'Top-3 Acc':<12} | {'Entropy (Bits)':<14}")
    print("-" * 65)
    for step in sim["trajectory"]:
        print(
            f"{step['label']:<22} | "
            f"{step['top_1_accuracy'] * 100:>10.1f}% | "
            f"{step['top_3_accuracy'] * 100:>10.1f}% | "
            f"{step['mean_entropy']:>14.3f}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_evaluation()
