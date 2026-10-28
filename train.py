"""
MedQuery: Training and Artifact Build Pipeline
Executes preprocessing, trains differential diagnosis model, runs benchmarks, and saves artifacts.
"""

import json
from pathlib import Path
import pandas as pd
from src.adaptive_questioner import AdaptiveQuestioner
from src.data_processor import DataProcessor
from src.diagnostic_engine import DiagnosticEngine
from src.evaluation import MedQueryEvaluator
from src.explainability import ExplainabilityEngine
from src.knowledge_base import MedicalKnowledgeBase


def run_pipeline():
    print("=" * 70)
    print(" MedQuery: Training and Evaluation Pipeline")
    print(" Adaptive Evidence-Guided Framework for Differential Diagnosis")
    print("=" * 70)

    # 1. Preprocess raw data
    print("\n[1/5] Processing raw datasets...")
    processor = DataProcessor(data_dir="data")
    train_df, test_df = processor.process_and_save()
    metadata = processor.load_metadata()

    symptom_columns = metadata["symptom_columns"]
    disease_classes = metadata["disease_classes"]
    symptom_display_map = metadata["symptom_display_map"]
    disease_profile = metadata["disease_profile"]

    print(f"  Processed {len(train_df)} training samples across {len(disease_classes)} diseases.")
    print(f"  Total distinct clinical symptoms: {len(symptom_columns)}")
    print(f"  Held-out test samples: {len(test_df)}")

    # 2. Train diagnostic engine
    print("\n[2/5] Training calibrated differential diagnostic ensemble...")
    engine = DiagnosticEngine(symptom_columns, disease_classes)
    train_metrics = engine.train(train_df, train_df["prognosis"])
    print(f"  Ensemble training complete. Train Accuracy: {train_metrics['rf_train_accuracy'] * 100:.2f}%")

    # 3. Save model
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "medquery_model.pkl"
    engine.save(str(model_path))
    print(f"  Model artifact saved to {model_path}")

    # 4. Initialize supporting engines
    print("\n[3/5] Initializing Medical Knowledge Base & Adaptive Questioner...")
    kb = MedicalKnowledgeBase(profiles=disease_profile)
    questioner = AdaptiveQuestioner(
        symptom_columns=symptom_columns,
        symptom_display_map=symptom_display_map,
        disease_profile=disease_profile,
        confidence_threshold=0.75,
        min_probability_gap=0.25,
        max_questions=5
    )
    explain_engine = ExplainabilityEngine(
        disease_profile=disease_profile,
        symptom_display_map=symptom_display_map,
        knowledge_base=kb
    )

    # 5. Run test set evaluation
    print("\n[4/5] Evaluating on held-out test dataset...")
    evaluator = MedQueryEvaluator(engine, questioner, symptom_columns)
    test_results = evaluator.evaluate_test_set(test_df)

    print("  --- Test Set Performance ---")
    print(f"  Top-1 Accuracy : {test_results['top_1_accuracy'] * 100:.2f}%")
    print(f"  Top-3 Accuracy : {test_results['top_3_accuracy'] * 100:.2f}%")
    print(f"  Top-5 Accuracy : {test_results['top_5_accuracy'] * 100:.2f}%")
    print(f"  Macro Precision: {test_results['macro_precision'] * 100:.2f}%")
    print(f"  Macro Recall   : {test_results['macro_recall'] * 100:.2f}%")
    print(f"  Macro F1-Score : {test_results['macro_f1'] * 100:.2f}%")

    # 6. Simulate Adaptive Questioning vs. Static Incomplete Presentation
    print("\n[5/5] Running Adaptive Questioning vs. Incomplete Presentation Experiment...")
    sim_results = evaluator.simulate_adaptive_questioning_experiment(
        test_df, num_initial_symptoms=1, max_followups=4
    )

    print("\n  Turn-by-Turn Dynamic Progression (Initial 1 symptom):")
    print(f"  {'Step':<20} | {'Top-1 Acc':<10} | {'Top-3 Acc':<10} | {'Mean Entropy':<12}")
    print("  " + "-" * 60)
    for step in sim_results["trajectory"]:
        print(
            f"  {step['label']:<20} | "
            f"{step['top_1_accuracy'] * 100:>8.1f}% | "
            f"{step['top_3_accuracy'] * 100:>8.1f}% | "
            f"{step['mean_entropy']:>10.3f}"
        )

    print(
        f"\n  Adaptive Gain: Top-1 improved from {sim_results['baseline_top1'] * 100:.1f}% "
        f"to {sim_results['final_adaptive_top1'] * 100:.1f}% "
        f"(+{sim_results['accuracy_gain_top1'] * 100:.1f}%)"
    )

    # Save full evaluation summary
    eval_summary = {
        "standard_test_metrics": {
            k: v for k, v in test_results.items() if k != "classification_report"
        },
        "adaptive_simulation": sim_results
    }
    with open("data/processed/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print("\nEvaluation artifacts saved to data/processed/evaluation_results.json")
    print("Pipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()
