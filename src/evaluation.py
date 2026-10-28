"""
MedQuery: Evaluation and Benchmarking Module
Computes Top-1, Top-3, Top-5 accuracy, precision, recall, F1,
and runs incomplete-information adaptive questioning simulations.
"""

import json
import warnings
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")


class MedQueryEvaluator:
    """Evaluates MedQuery diagnostic models against standard and conversational benchmarks."""

    def __init__(self, diagnostic_engine, adaptive_questioner, symptom_columns: List[str]):
        self.diagnostic_engine = diagnostic_engine
        self.adaptive_questioner = adaptive_questioner
        self.symptom_columns = symptom_columns

    def evaluate_test_set(self, test_df: pd.DataFrame) -> Dict:
        """
        Calculates standard classification metrics: Top-1, Top-3, Top-5 accuracy,
        precision, recall, and F1-score on the held-out test dataset.
        """
        y_true = []
        y_pred_top1 = []
        top3_hits = []
        top5_hits = []

        for _, row in test_df.iterrows():
            true_disease = row["prognosis"]
            present_symptoms = [c for c in self.symptom_columns if row[c] == 1]

            pred_res = self.diagnostic_engine.predict_differential(
                present_symptoms=present_symptoms,
                top_k=5
            )
            differential = pred_res["differential"]
            predicted_names = [item["disease"] for item in differential]

            y_true.append(true_disease)
            y_pred_top1.append(predicted_names[0] if predicted_names else "")
            top3_hits.append(true_disease in predicted_names[:3])
            top5_hits.append(true_disease in predicted_names[:5])

        top1_acc = float(np.mean([t == p for t, p in zip(y_true, y_pred_top1)]))
        top3_acc = float(np.mean(top3_hits))
        top5_acc = float(np.mean(top5_hits))

        macro_precision = float(precision_score(y_true, y_pred_top1, average="macro", zero_division=0))
        macro_recall = float(recall_score(y_true, y_pred_top1, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred_top1, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred_top1, average="weighted", zero_division=0))

        report_dict = classification_report(y_true, y_pred_top1, output_dict=True, zero_division=0)

        results = {
            "top_1_accuracy": round(top1_acc, 4),
            "top_3_accuracy": round(top3_acc, 4),
            "top_5_accuracy": round(top5_acc, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "sample_count": len(test_df),
            "classification_report": report_dict
        }
        return results

    def simulate_adaptive_questioning_experiment(
        self,
        test_df: pd.DataFrame,
        num_initial_symptoms: int = 1,
        max_followups: int = 4
    ) -> Dict:
        """
        Simulates clinical scenario with incomplete initial patient information:
        Compares static baseline (no questioning) against MedQuery adaptive questioning steps.
        """
        np.random.seed(42)
        total_samples = len(test_df)

        # Track accuracy at turn 0 (initial incomplete input) and subsequent turns 1..max_followups
        turn_top1_hits = {turn: 0 for turn in range(max_followups + 1)}
        turn_top3_hits = {turn: 0 for turn in range(max_followups + 1)}
        turn_entropy = {turn: [] for turn in range(max_followups + 1)}

        for _, row in test_df.iterrows():
            true_disease = row["prognosis"]
            all_true_symptoms = set([c for c in self.symptom_columns if row[c] == 1])

            if not all_true_symptoms:
                continue

            # Pick a subset of true symptoms as initial patient presentation
            symptom_list = list(all_true_symptoms)
            k = min(num_initial_symptoms, len(symptom_list))
            initial_symptoms = list(np.random.choice(symptom_list, size=k, replace=False))

            current_present = list(initial_symptoms)
            current_denied = []
            asked_set = set(initial_symptoms)

            # Turn 0: Baseline static prediction with incomplete info
            res0 = self.diagnostic_engine.predict_differential(current_present, current_denied, top_k=5)
            d0 = [x["disease"] for x in res0["differential"]]
            if d0 and d0[0] == true_disease:
                turn_top1_hits[0] += 1
            if true_disease in d0[:3]:
                turn_top3_hits[0] += 1
            turn_entropy[0].append(res0.get("entropy", 0.0))

            # Adaptive follow-up turns 1 to max_followups
            for turn in range(1, max_followups + 1):
                next_q = self.adaptive_questioner.select_next_question(
                    res0, asked_set, question_count=turn - 1
                )

                if next_q is not None:
                    sym = next_q["symptom_key"]
                    asked_set.add(sym)

                    # Simulated patient response against ground truth
                    if sym in all_true_symptoms:
                        current_present.append(sym)
                    else:
                        current_denied.append(sym)

                # Recompute differential with newly collected evidence
                res0 = self.diagnostic_engine.predict_differential(current_present, current_denied, top_k=5)
                d_turn = [x["disease"] for x in res0["differential"]]

                if d_turn and d_turn[0] == true_disease:
                    turn_top1_hits[turn] += 1
                if true_disease in d_turn[:3]:
                    turn_top3_hits[turn] += 1
                turn_entropy[turn].append(res0.get("entropy", 0.0))

        # Compile trajectory metrics
        trajectory = []
        for turn in range(max_followups + 1):
            top1_acc = turn_top1_hits[turn] / total_samples
            top3_acc = turn_top3_hits[turn] / total_samples
            avg_entropy = float(np.mean(turn_entropy[turn])) if turn_entropy[turn] else 0.0

            trajectory.append({
                "followup_turn": turn,
                "label": "Initial Incomplete" if turn == 0 else f"After Q{turn}",
                "top_1_accuracy": round(top1_acc, 4),
                "top_3_accuracy": round(top3_acc, 4),
                "mean_entropy": round(avg_entropy, 3)
            })

        return {
            "initial_symptom_count": num_initial_symptoms,
            "sample_count": total_samples,
            "trajectory": trajectory,
            "baseline_top1": trajectory[0]["top_1_accuracy"],
            "final_adaptive_top1": trajectory[-1]["top_1_accuracy"],
            "accuracy_gain_top1": round(trajectory[-1]["top_1_accuracy"] - trajectory[0]["top_1_accuracy"], 4),
            "baseline_top3": trajectory[0]["top_3_accuracy"],
            "final_adaptive_top3": trajectory[-1]["top_3_accuracy"],
            "accuracy_gain_top3": round(trajectory[-1]["top_3_accuracy"] - trajectory[0]["top_3_accuracy"], 4)
        }
