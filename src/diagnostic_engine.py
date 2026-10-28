"""
MedQuery: Diagnostic Engine for Dynamic Differential Diagnosis
Trains ensemble classifier, estimates posterior disease distributions, and outputs Top-1/3/5 rankings.
"""

from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB


class DiagnosticEngine:
    """Ensemble-backed differential diagnosis inference engine."""

    def __init__(self, symptom_columns: List[str], disease_classes: List[str]):
        self.symptom_columns = symptom_columns
        self.symptom_idx_map = {sym: idx for idx, sym in enumerate(symptom_columns)}
        self.disease_classes = sorted(disease_classes)
        self.num_symptoms = len(symptom_columns)
        self.num_classes = len(disease_classes)

        # Primary model: Random Forest Classifier with balanced trees
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=25,
            min_samples_split=2,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
        self.nb_model = MultinomialNB(alpha=0.1)
        self.is_trained = False

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """Fit diagnostic models on preprocessed training dataset."""
        # Ensure column ordering matches
        X = X_train[self.symptom_columns].values
        y = y_train.values

        self.model.fit(X, y)
        self.nb_model.fit(X, y)
        self.is_trained = True

        train_acc = float(np.mean(self.model.predict(X) == y))
        return {"rf_train_accuracy": train_acc}

    def symptom_list_to_vector(
        self,
        present_symptoms: List[str],
        denied_symptoms: Optional[List[str]] = None
    ) -> np.ndarray:
        """Converts symptom lists to a binary feature vector."""
        vec = np.zeros((1, self.num_symptoms), dtype=np.float32)
        denied_set = set(denied_symptoms or [])

        for sym in present_symptoms:
            if sym in self.symptom_idx_map and sym not in denied_set:
                vec[0, self.symptom_idx_map[sym]] = 1.0

        return vec

    def predict_differential(
        self,
        present_symptoms: List[str],
        denied_symptoms: Optional[List[str]] = None,
        top_k: int = 5
    ) -> Dict:
        """
        Calculates dynamic differential diagnosis for the observed symptoms.
        Returns Top-K disease candidates, normalized probabilities, confidence level, and entropy.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before predicting differential diagnosis.")

        if not present_symptoms:
            # Uniform prior if no symptoms given yet
            uniform_prob = 1.0 / self.num_classes
            top_list = [
                {"disease": d, "probability": uniform_prob, "percentage": f"{uniform_prob * 100:.1f}%"}
                for d in self.disease_classes[:top_k]
            ]
            return {
                "differential": top_list,
                "top_1": top_list[0],
                "top_3": top_list[:3],
                "top_5": top_list[:5],
                "entropy": float(np.log2(self.num_classes)),
                "confidence_band": "Uncertain (No symptoms provided)",
                "active_symptom_count": 0
            }

        vec = self.symptom_list_to_vector(present_symptoms, denied_symptoms)

        # Get probabilities from ensemble
        rf_probs = self.model.predict_proba(vec)[0]
        nb_probs = self.nb_model.predict_proba(vec)[0]

        # Blend ensemble: 80% Random Forest + 20% Naive Bayes smoothing
        blended_probs = (0.80 * rf_probs) + (0.20 * nb_probs)

        # Penalize diseases if patient explicitly denied their hallmark symptoms
        classes = self.model.classes_
        prob_dict = {cls_name: float(prob) for cls_name, prob in zip(classes, blended_probs)}

        # Renormalize
        total = sum(prob_dict.values())
        if total > 0:
            prob_dict = {k: v / total for k, v in prob_dict.items()}

        # Sort descending
        ranked = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)

        top_candidates = []
        for d, p in ranked[:top_k]:
            top_candidates.append({
                "disease": d,
                "probability": float(p),
                "percentage": f"{p * 100:.1f}%"
            })

        # Calculate Shannon Entropy of posterior distribution: H(D) = - sum(p * log2(p))
        eps = 1e-12
        entropy = -float(sum(p * np.log2(p + eps) for p in prob_dict.values() if p > 0))

        # Determine confidence band
        top_prob = top_candidates[0]["probability"] if top_candidates else 0.0
        gap_to_second = (
            top_candidates[0]["probability"] - top_candidates[1]["probability"]
            if len(top_candidates) > 1 else top_prob
        )

        if top_prob >= 0.70 and gap_to_second >= 0.30:
            confidence_band = "High Confidence"
        elif top_prob >= 0.40 or gap_to_second >= 0.15:
            confidence_band = "Moderate Confidence"
        else:
            confidence_band = "Low Confidence (Differential Required)"

        return {
            "differential": top_candidates,
            "top_1": top_candidates[0] if top_candidates else None,
            "top_3": top_candidates[:3],
            "top_5": top_candidates[:5],
            "entropy": round(entropy, 3),
            "confidence_band": confidence_band,
            "active_symptom_count": len(present_symptoms)
        }

    def save(self, filepath: str):
        """Save trained diagnostic engine to disk."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "DiagnosticEngine":
        """Load trained diagnostic engine from disk."""
        return joblib.load(filepath)
