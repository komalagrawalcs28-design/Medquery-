"""
MedQuery: Explainable AI & Evidence Attribution Engine
Generates clinical explanations, supporting vs missing symptoms, and uncertainty metrics.
"""

from typing import Dict, List, Optional
import numpy as np


class ExplainabilityEngine:
    """Computes transparent clinical evidence breakdowns and uncertainty metrics."""

    def __init__(
        self,
        disease_profile: Dict[str, Dict],
        symptom_display_map: Dict[str, str],
        knowledge_base
    ):
        self.disease_profile = disease_profile
        self.symptom_display_map = symptom_display_map
        self.knowledge_base = knowledge_base

    def explain_disease(
        self,
        disease_name: str,
        probability: float,
        present_symptoms: List[str],
        denied_symptoms: Optional[List[str]] = None
    ) -> Dict:
        """Constructs an evidence breakdown for a single candidate disease."""
        denied_set = set(denied_symptoms or [])
        present_set = set(present_symptoms)

        profile = self.disease_profile.get(disease_name, {})
        hallmarks = profile.get("hallmark_symptoms", [])
        freq_map = profile.get("symptom_frequencies", {})

        # Supporting evidence: patient has symptom AND it has high correlation with disease
        supporting = []
        for sym in present_symptoms:
            freq = freq_map.get(sym, 0.0)
            if freq >= 0.4:
                supporting.append({
                    "symptom_key": sym,
                    "display_name": self.symptom_display_map.get(sym, sym),
                    "disease_frequency": f"{freq * 100:.0f}%",
                    "importance": "Major Hallmark" if freq >= 0.8 else "Common Feature"
                })

        # Missing hallmark symptoms: hallmark symptoms the patient did NOT report
        missing = []
        for sym in hallmarks:
            if sym not in present_set and sym not in denied_set:
                missing.append({
                    "symptom_key": sym,
                    "display_name": self.symptom_display_map.get(sym, sym),
                    "typical_frequency": f"{freq_map.get(sym, 0.0) * 100:.0f}%"
                })

        # Contradictory / Denied symptoms: hallmark symptoms the patient explicitly denied
        conflicts = []
        for sym in hallmarks:
            if sym in denied_set:
                conflicts.append({
                    "symptom_key": sym,
                    "display_name": self.symptom_display_map.get(sym, sym),
                    "typical_frequency": f"{freq_map.get(sym, 0.0) * 100:.0f}%"
                })

        # Match ratio
        match_score = (len(supporting) / len(hallmarks)) if hallmarks else 0.0

        # Knowledge base clinical details
        kb_info = self.knowledge_base.get_disease_info(disease_name)

        return {
            "disease": disease_name,
            "probability": probability,
            "percentage": f"{probability * 100:.1f}%",
            "evidence_match_score": round(match_score, 2),
            "supporting_symptoms": supporting,
            "missing_hallmarks": missing,
            "denied_conflicts": conflicts,
            "urgency": kb_info.get("urgency", "Moderate"),
            "specialist": kb_info.get("specialist", "General Physician"),
            "description": kb_info.get("description", ""),
            "precautions": kb_info.get("precautions", []),
            "warning_signs": kb_info.get("warning_signs", []),
        }

    def generate_full_explanation(
        self,
        differential_result: Dict,
        present_symptoms: List[str],
        denied_symptoms: Optional[List[str]] = None
    ) -> Dict:
        """Generates comprehensive differential explanation report across Top-3/Top-5 candidates."""
        candidates = differential_result.get("differential", [])
        detailed_candidates = []

        for item in candidates:
            d_name = item["disease"]
            d_prob = item["probability"]
            exp = self.explain_disease(d_name, d_prob, present_symptoms, denied_symptoms)
            detailed_candidates.append(exp)

        # Global uncertainty analysis
        entropy = differential_result.get("entropy", 0.0)
        top_prob = candidates[0]["probability"] if candidates else 0.0
        gap = (
            candidates[0]["probability"] - candidates[1]["probability"]
            if len(candidates) > 1 else top_prob
        )

        if entropy < 1.0 and top_prob >= 0.70:
            uncertainty_text = "Low diagnostic uncertainty: Symptoms closely match a specific clinical entity."
            uncertainty_tag = "Low"
        elif entropy < 2.0 or gap >= 0.20:
            uncertainty_text = "Moderate diagnostic uncertainty: Plausible primary entity with a few secondary differentials."
            uncertainty_tag = "Moderate"
        else:
            uncertainty_text = "High diagnostic uncertainty: Broad differential diagnosis; additional clinical tests required."
            uncertainty_tag = "High"

        # Check if any top candidates have emergency triage status
        emergency_flags = [c for c in detailed_candidates[:3] if c["urgency"] == "Emergency"]

        return {
            "candidates": detailed_candidates,
            "top_candidate": detailed_candidates[0] if detailed_candidates else None,
            "uncertainty_level": uncertainty_tag,
            "uncertainty_description": uncertainty_text,
            "entropy": entropy,
            "probability_gap": round(gap, 3),
            "emergency_alert": len(emergency_flags) > 0,
            "emergency_conditions": [c["disease"] for c in emergency_flags]
        }
