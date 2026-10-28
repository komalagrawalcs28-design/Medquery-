"""
MedQuery: Adaptive Follow-up Questioning Engine
Selects the most discriminative missing symptoms using Information Gain and entropy reduction.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np


# Question phrasing templates for common symptoms
SYMPTOM_QUESTION_TEMPLATES: Dict[str, str] = {
    "high_fever": "Are you running a high fever or feeling unusually hot?",
    "mild_fever": "Do you have a mild or low-grade fever?",
    "chills": "Are you experiencing chills or feeling sudden coldness?",
    "shivering": "Have you noticed uncontrollable shivering or shaking?",
    "headache": "Do you have a headache or pain in your head/forehead?",
    "vomiting": "Have you had any vomiting or throwing up?",
    "nausea": "Are you feeling nauseous or queasy in your stomach?",
    "cough": "Do you have a persistent cough?",
    "breathlessness": "Are you experiencing shortness of breath or difficulty breathing?",
    "chest_pain": "Are you feeling any chest pain, tightness, or heaviness in your chest?",
    "fatigue": "Have you been experiencing significant fatigue, exhaustion, or weakness?",
    "skin_rash": "Have you noticed any skin rash or redness on your body?",
    "itching": "Do you have persistent skin itching or irritation?",
    "joint_pain": "Are you suffering from aching or pain in your joints?",
    "stomach_pain": "Do you have stomach pain or abdominal discomfort?",
    "abdominal_pain": "Are you feeling pain or cramps in your abdomen?",
    "acidity": "Have you been experiencing acidity, acid reflux, or heartburn?",
    "diarrhoea": "Have you had loose, frequent, or watery bowel movements (diarrhea)?",
    "constipation": "Are you experiencing difficulty with bowel movements (constipation)?",
    "dizziness": "Do you feel lightheaded, faint, or dizzy?",
    "loss_of_balance": "Have you felt unsteadiness or difficulty maintaining balance?",
    "yellowish_skin": "Have you noticed a yellowish tint to your skin or pale appearance?",
    "yellowing_of_eyes": "Have you or others noticed any yellowing in the whites of your eyes?",
    "dark_urine": "Is your urine darker than usual (tea-colored or deep amber)?",
    "burning_micturition": "Do you feel a burning sensation or pain while urinating?",
    "polyuria": "Are you needing to urinate much more frequently than normal?",
    "continuous_sneezing": "Have you had frequent or continuous bouts of sneezing?",
    "runny_nose": "Do you have a runny or congested nose?",
    "throat_irritation": "Do you feel irritation, soreness, or scratchiness in your throat?",
    "loss_of_appetite": "Have you experienced a noticeable loss of appetite or dislike for food?",
    "excessive_hunger": "Are you experiencing abnormally increased appetite or excessive hunger?",
    "weight_loss": "Have you noticed any recent unintentional weight loss?",
    "sweating": "Have you been having excessive sweating or night sweats?",
    "cold_hands_and_feets": "Do your hands or feet feel unusually cold or clammy?",
    "fast_heart_rate": "Do you feel your heart racing, pounding, or beating unusually fast?",
    "palpitations": "Are you aware of rapid, fluttering, or skipping heartbeats?",
    "muscle_pain": "Are you experiencing general muscle aches or body pain?",
    "back_pain": "Do you have lower or upper back pain?",
    "neck_pain": "Are you suffering from neck stiffness or neck pain?",
}


def generate_question_for_symptom(symptom_key: str, display_name: str) -> str:
    """Returns a natural phrasing question for a given symptom."""
    if symptom_key in SYMPTOM_QUESTION_TEMPLATES:
        return SYMPTOM_QUESTION_TEMPLATES[symptom_key]
    return f"Are you currently experiencing any {display_name.lower()}?"


class AdaptiveQuestioner:
    """Selects optimal discriminative questions to narrow differential diagnosis."""

    def __init__(
        self,
        symptom_columns: List[str],
        symptom_display_map: Dict[str, str],
        disease_profile: Dict[str, Dict],
        confidence_threshold: float = 0.75,
        min_probability_gap: float = 0.25,
        max_questions: int = 5
    ):
        self.symptom_columns = symptom_columns
        self.symptom_display_map = symptom_display_map
        self.disease_profile = disease_profile
        self.confidence_threshold = confidence_threshold
        self.min_probability_gap = min_probability_gap
        self.max_questions = max_questions

    def get_candidate_information_gain(
        self,
        candidate_diseases: List[Dict],
        asked_symptoms: Set[str]
    ) -> List[Tuple[str, float]]:
        """
        Calculates information gain / discriminative variance for each unasked symptom
        conditioned on current top candidate disease probabilities.
        """
        if not candidate_diseases:
            return []

        # Extract candidates and their posterior weights
        top_diseases = candidate_diseases[:5]
        probs = np.array([c["probability"] for c in top_diseases], dtype=np.float32)
        total_p = probs.sum()
        if total_p > 0:
            probs = probs / total_p
        else:
            probs = np.ones(len(top_diseases)) / len(top_diseases)

        symptom_scores = []

        for sym in self.symptom_columns:
            if sym in asked_symptoms:
                continue

            # Gather frequency of this symptom across top candidates
            freqs = []
            for c in top_diseases:
                d_name = c["disease"]
                d_info = self.disease_profile.get(d_name, {})
                f = d_info.get("symptom_frequencies", {}).get(sym, 0.0)
                freqs.append(f)

            freq_arr = np.array(freqs, dtype=np.float32)

            # Weighted mean frequency: E[s] = sum(p_i * f_i)
            mean_f = np.sum(probs * freq_arr)

            # Weighted variance: Var(s) = sum(p_i * (f_i - mean_f)^2)
            # High variance means the symptom strongly separates the candidate diseases!
            var_s = np.sum(probs * (freq_arr - mean_f) ** 2)

            # Penalize symptoms with zero prevalence in top diseases
            if mean_f < 0.05:
                continue

            # Information Gain score combines variance with relevance
            score = float(var_s * (1.0 + mean_f))
            if score > 1e-4:
                symptom_scores.append((sym, score))

        # Sort descending by discriminative score
        symptom_scores.sort(key=lambda x: x[1], reverse=True)
        return symptom_scores

    def select_next_question(
        self,
        differential_result: Dict,
        asked_symptoms: Set[str],
        question_count: int
    ) -> Optional[Dict]:
        """
        Determines whether to ask a follow-up question and selects the best candidate symptom.
        Returns question dict if follow-up is needed, or None if stopping criteria met.
        """
        # Stop condition 1: reached maximum allowed questions
        if question_count >= self.max_questions:
            return None

        differential = differential_result.get("differential", [])
        if not differential:
            return None

        top_prob = differential[0]["probability"]
        second_prob = differential[1]["probability"] if len(differential) > 1 else 0.0
        gap = top_prob - second_prob

        # Stop condition 2: high confidence reached with substantial differential gap
        if top_prob >= self.confidence_threshold and gap >= self.min_probability_gap:
            return None

        # Calculate scores for unasked symptoms
        scores = self.get_candidate_information_gain(differential, asked_symptoms)
        if not scores:
            return None

        best_symptom, best_score = scores[0]
        display_name = self.symptom_display_map.get(best_symptom, best_symptom.replace("_", " ").title())
        question_text = generate_question_for_symptom(best_symptom, display_name)

        return {
            "symptom_key": best_symptom,
            "display_name": display_name,
            "question_text": question_text,
            "discriminative_score": round(best_score, 4),
            "question_index": question_count + 1,
            "total_questions_allowed": self.max_questions,
            "differentiating_between": [c["disease"] for c in differential[:3]]
        }
