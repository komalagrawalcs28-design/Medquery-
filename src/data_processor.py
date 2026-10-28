"""
MedQuery: Data Preprocessing and Feature Engineering
Cleans raw symptom datasets, standardizes column formats, and computes symptom profiles.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


# Disease name canonicalization dictionary
DISEASE_NAME_MAP = {
    "Diabetes ": "Diabetes",
    "Peptic ulcer diseae": "Peptic Ulcer Disease",
    "Osteoarthristis": "Osteoarthritis",
    "hepatitis A": "Hepatitis A",
    "(vertigo) Paroymsal  Positional Vertigo": "Paroxysmal Positional Vertigo",
    "Dimorphic hemmorhoids(piles)": "Hemorrhoids (Piles)",
    "GERD": "Gastroesophageal Reflux Disease (GERD)",
    "AIDS": "AIDS / HIV",
    "Urinary tract infection": "Urinary Tract Infection (UTI)",
    "Varicose veins": "Varicose Veins",
    "Fungal infection": "Fungal Infection",
    "Drug Reaction": "Drug Reaction",
    "Chronic cholestasis": "Chronic Cholestasis",
    "Bronchial Asthma": "Bronchial Asthma",
    "Cervical spondylosis": "Cervical Spondylosis",
    "Common Cold": "Common Cold",
    "Heart attack": "Heart Attack",
    "Paralysis (brain hemorrhage)": "Paralysis (Brain Hemorrhage)",
}

# Symptom column name standardization
COLUMN_RENAME_MAP = {
    "spotting_ urination": "spotting_urination",
    "foul_smell_of urine": "foul_smell_of_urine",
    "dischromic _patches": "dischromic_patches",
    "toxic_look_(typhos)": "toxic_look_typhos",
}


def clean_symptom_key(col: str) -> str:
    """Format raw column name to standard snake_case symptom key."""
    col = col.strip()
    if col in COLUMN_RENAME_MAP:
        col = COLUMN_RENAME_MAP[col]
    return col


def format_symptom_display(symptom_key: str) -> str:
    """Convert snake_case key to human-readable capitalized display label."""
    display = symptom_key.replace("_", " ").title()
    # Specialized acronym / medical terms capitalization
    replacements = {
        "Gerd": "GERD",
        "Aids": "AIDS",
        "Uti": "UTI",
        "Cold Hands And Feets": "Cold Hands and Feet",
        "Patches In Throat": "Throat Patches",
        "Toxic Look Typhos": "Toxic Appearance (Typhoid Look)",
        "Dischromic Patches": "Dischromic Skin Patches",
        "Spotting Urination": "Spotting During Urination",
        "Foul Smell Of Urine": "Foul-Smelling Urine",
        "Continuous Feel Of Urine": "Continuous Urge to Urinate",
    }
    return replacements.get(display, display)


class DataProcessor:
    """Handles data loading, cleaning, validation, and feature metadata generation."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.symptom_columns: List[str] = []
        self.disease_classes: List[str] = []
        self.symptom_display_map: Dict[str, str] = {}
        self.disease_profile: Dict[str, Dict] = {}

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize raw dataframe columns, merge duplicates, and clean targets."""
        df = df.copy()

        # Drop spurious trailing unamed column if present
        unnamed_cols = [c for c in df.columns if "Unnamed" in str(c)]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)

        # Standardize column names
        rename_dict = {c: clean_symptom_key(c) for c in df.columns if c != "prognosis"}
        df = df.rename(columns=rename_dict)

        # Handle fluid_overload and fluid_overload.1 collision
        if "fluid_overload" in df.columns and "fluid_overload.1" in df.columns:
            df["fluid_overload"] = (df["fluid_overload"] | df["fluid_overload.1"]).astype(int)
            df = df.drop(columns=["fluid_overload.1"])

        # Clean target prognosis label
        if "prognosis" in df.columns:
            df["prognosis"] = df["prognosis"].astype(str).str.strip()
            df["prognosis"] = df["prognosis"].apply(
                lambda x: DISEASE_NAME_MAP.get(x, x.title() if not x.isupper() else x)
            )

        return df

    def process_and_save(
        self,
        raw_train_file: str = "Training.csv",
        raw_test_file: str = "Testing.csv",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads raw files, applies cleaning pipeline, builds profile metadata, and saves."""
        train_path = self.raw_dir / raw_train_file
        test_path = self.raw_dir / raw_test_file

        if not train_path.exists():
            raise FileNotFoundError(f"Training dataset not found at {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Testing dataset not found at {test_path}")

        raw_train = pd.read_csv(train_path)
        raw_test = pd.read_csv(test_path)

        train_clean = self.clean_dataframe(raw_train)
        test_clean = self.clean_dataframe(raw_test)

        # Align columns
        symptom_cols = [c for c in train_clean.columns if c != "prognosis"]
        test_clean = test_clean[symptom_cols + ["prognosis"]]

        self.symptom_columns = symptom_cols
        self.disease_classes = sorted(train_clean["prognosis"].unique().tolist())
        self.symptom_display_map = {col: format_symptom_display(col) for col in symptom_cols}

        # Build empirical symptom conditional probability profiles P(Symptom=1 | Disease)
        profiles = {}
        for disease, group in train_clean.groupby("prognosis"):
            symptom_freq = group[symptom_cols].mean().to_dict()
            hallmarks = [s for s, freq in symptom_freq.items() if freq >= 0.5]
            profiles[disease] = {
                "sample_count": int(len(group)),
                "symptom_frequencies": symptom_freq,
                "hallmark_symptoms": hallmarks,
            }
        self.disease_profile = profiles

        # Save processed CSVs
        train_clean.to_csv(self.processed_dir / "cleaned_train.csv", index=False)
        test_clean.to_csv(self.processed_dir / "cleaned_test.csv", index=False)

        # Save metadata lexicon
        metadata = {
            "symptom_columns": self.symptom_columns,
            "disease_classes": self.disease_classes,
            "symptom_display_map": self.symptom_display_map,
            "disease_profile": self.disease_profile,
        }
        with open(self.processed_dir / "symptom_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return train_clean, test_clean

    def load_metadata(self) -> Dict:
        """Load pre-computed metadata."""
        meta_file = self.processed_dir / "symptom_metadata.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata not found at {meta_file}. Run process_and_save first.")
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)
