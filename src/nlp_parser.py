"""
MedQuery: Natural Language Symptom Parser
Extracts positive and negated symptom mentions from conversational free-text inputs.
"""

import re
from typing import Dict, List, Set, Tuple


# Synonym and alias dictionary mapping layman terms to canonical symptom keys
SYMPTOM_SYNONYM_MAP: Dict[str, str] = {
    # General / Systemic
    "fever": "high_fever",
    "high fever": "high_fever",
    "mild fever": "mild_fever",
    "low grade fever": "mild_fever",
    "temperature": "high_fever",
    "feverish": "high_fever",
    "chills": "chills",
    "shivering": "shivering",
    "shivers": "shivering",
    "sweating": "sweating",
    "sweats": "sweating",
    "night sweats": "sweating",
    "fatigue": "fatigue",
    "tired": "fatigue",
    "tiredness": "fatigue",
    "exhaustion": "fatigue",
    "exhausted": "fatigue",
    "weakness": "fatigue",
    "lethargy": "lethargy",
    "malaise": "malaise",
    "unwell": "malaise",
    "restlessness": "restlessness",
    "anxiety": "anxiety",
    "anxious": "anxiety",
    "weight loss": "weight_loss",
    "losing weight": "weight_loss",
    "weight gain": "weight_gain",
    "dehydration": "dehydration",
    "dehydrated": "dehydration",

    # Neurological / Pain
    "headache": "headache",
    "head ache": "headache",
    "head hurting": "headache",
    "migraine": "headache",
    "dizziness": "dizziness",
    "dizzy": "dizziness",
    "lightheaded": "dizziness",
    "spinning": "spinning_movements",
    "vertigo": "spinning_movements",
    "loss of balance": "loss_of_balance",
    "unsteady": "unsteadiness",
    "unsteadiness": "unsteadiness",
    "slurred speech": "slurred_speech",
    "stiff neck": "stiff_neck",
    "neck pain": "neck_pain",
    "back pain": "back_pain",
    "altered sensorium": "altered_sensorium",
    "confusion": "altered_sensorium",
    "lack of concentration": "lack_of_concentration",

    # Respiratory / ENT
    "cough": "cough",
    "coughing": "cough",
    "sneezing": "continuous_sneezing",
    "continuous sneezing": "continuous_sneezing",
    "runny nose": "runny_nose",
    "congestion": "congestion",
    "stuffy nose": "congestion",
    "sinus pressure": "sinus_pressure",
    "breathlessness": "breathlessness",
    "short of breath": "breathlessness",
    "shortness of breath": "breathlessness",
    "difficulty breathing": "breathlessness",
    "trouble breathing": "breathlessness",
    "phlegm": "phlegm",
    "mucus": "mucoid_sputum",
    "sputum": "mucoid_sputum",
    "blood in sputum": "blood_in_sputum",
    "coughing blood": "blood_in_sputum",
    "throat irritation": "throat_irritation",
    "sore throat": "throat_irritation",
    "patches in throat": "patches_in_throat",
    "chest pain": "chest_pain",
    "chest tightness": "chest_pain",

    # Gastrointestinal
    "stomach pain": "stomach_pain",
    "stomach ache": "stomach_pain",
    "tummy ache": "stomach_pain",
    "belly pain": "belly_pain",
    "abdominal pain": "abdominal_pain",
    "cramps": "cramps",
    "nausea": "nausea",
    "nauseous": "nausea",
    "queasy": "nausea",
    "vomiting": "vomiting",
    "vomit": "vomiting",
    "throwing up": "vomiting",
    "puking": "vomiting",
    "acidity": "acidity",
    "heartburn": "acidity",
    "acid reflux": "acidity",
    "indigestion": "indigestion",
    "diarrhea": "diarrhoea",
    "diarrhoea": "diarrhoea",
    "loose motion": "diarrhoea",
    "constipation": "constipation",
    "bloating": "distention_of_abdomen",
    "gas": "passage_of_gases",
    "loss of appetite": "loss_of_appetite",
    "no appetite": "loss_of_appetite",
    "excessive hunger": "excessive_hunger",
    "increased appetite": "increased_appetite",
    "stomach bleeding": "stomach_bleeding",
    "bloody stool": "bloody_stool",
    "blood in stool": "bloody_stool",
    "swelling of stomach": "swelling_of_stomach",

    # Dermatological
    "itching": "itching",
    "itchy": "itching",
    "itch": "itching",
    "skin rash": "skin_rash",
    "rash": "skin_rash",
    "red spots": "red_spots_over_body",
    "pimples": "pus_filled_pimples",
    "acne": "pus_filled_pimples",
    "blackheads": "blackheads",
    "blister": "blister",
    "blisters": "blister",
    "skin peeling": "skin_peeling",
    "flaking skin": "skin_peeling",
    "bruising": "bruising",
    "bruises": "bruising",
    "yellow skin": "yellowish_skin",
    "yellowing of skin": "yellowish_skin",
    "yellow eyes": "yellowing_of_eyes",
    "yellowing of eyes": "yellowing_of_eyes",
    "jaundice": "yellowing_of_eyes",

    # Musculoskeletal
    "joint pain": "joint_pain",
    "joints pain": "joint_pain",
    "aching joints": "joint_pain",
    "knee pain": "knee_pain",
    "hip pain": "hip_joint_pain",
    "muscle pain": "muscle_pain",
    "body ache": "muscle_pain",
    "body pain": "muscle_pain",
    "muscle weakness": "muscle_weakness",
    "swelling joints": "swelling_joints",
    "swollen joints": "swelling_joints",
    "painful walking": "painful_walking",
    "difficulty walking": "painful_walking",

    # Cardiovascular & Circulation
    "palpitations": "palpitations",
    "fast heart rate": "fast_heart_rate",
    "racing heart": "fast_heart_rate",
    "cold hands": "cold_hands_and_feets",
    "cold feet": "cold_hands_and_feets",
    "swollen legs": "swollen_legs",
    "swelling in legs": "swollen_legs",

    # Urinary / Excretory
    "burning urination": "burning_micturition",
    "burning when peeing": "burning_micturition",
    "frequent urination": "polyuria",
    "peeing a lot": "polyuria",
    "dark urine": "dark_urine",
    "yellow urine": "yellow_urine",
    "blood in urine": "spotting_urination",
    "bladder discomfort": "bladder_discomfort",
    "foul smelling urine": "foul_smell_of_urine",

    # Sensory / Eye
    "blurred vision": "blurred_and_distorted_vision",
    "blurry vision": "blurred_and_distorted_vision",
    "red eyes": "redness_of_eyes",
    "redness of eyes": "redness_of_eyes",
    "watering eyes": "watering_from_eyes",
    "watering from eyes": "watering_from_eyes",
    "pain behind eyes": "pain_behind_the_eyes",
    "loss of smell": "loss_of_smell",
}

NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bdenies\b",
    r"\bdenied\b",
    r"\bwithout\b",
    r"\bdon't have\b",
    r"\bdo not have\b",
    r"\bnever had\b",
    r"\babsent\b",
    r"\bfree of\b",
]


class SymptomNLPParser:
    """Extracts confirmed and negated symptoms from conversational user input."""

    def __init__(self, valid_symptom_keys: List[str]):
        self.valid_symptom_keys = set(valid_symptom_keys)
        # Build phrase dictionary sorted by longest phrase first
        self.phrase_map = {}
        for synonym, canonical in SYMPTOM_SYNONYM_MAP.items():
            if canonical in self.valid_symptom_keys:
                self.phrase_map[synonym.lower()] = canonical

        # Add all canonical keys and their space-separated versions
        for sym in valid_symptom_keys:
            phrase = sym.replace("_", " ").lower()
            self.phrase_map[phrase] = sym
            self.phrase_map[sym.lower()] = sym

        self.sorted_phrases = sorted(self.phrase_map.keys(), key=lambda x: len(x), reverse=True)

    def parse_text(self, text: str) -> Dict[str, Set[str]]:
        """Parses a free-text string and returns confirmed and negated symptom sets."""
        text = text.lower()
        confirmed = set()
        negated = set()

        # Split into sentences or clauses
        clauses = re.split(r"[,;.\n]|(?:\band\b)|(?:\bbut\b)", text)

        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue

            # Check if this clause contains negation
            is_negated = any(re.search(pat, clause) for pat in NEGATION_PATTERNS)

            for phrase in self.sorted_phrases:
                # Word boundary search for the phrase
                pattern = r"\b" + re.escape(phrase) + r"\b"
                if re.search(pattern, clause):
                    canonical_key = self.phrase_map[phrase]
                    if is_negated:
                        negated.add(canonical_key)
                    else:
                        confirmed.add(canonical_key)

        # Confirmed takes precedence if contradiction
        negated = negated - confirmed

        return {
            "confirmed": list(confirmed),
            "negated": list(negated),
        }
