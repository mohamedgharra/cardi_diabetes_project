"""Configuration constants for the cardio diabetes project."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import random

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
EDA_DIR = REPORTS_DIR / "eda"
MODELS_DIR = REPORTS_DIR / "models"
FIGS_DIR = REPORTS_DIR / "figs"
TABLES_DIR = REPORTS_DIR / "tables"
SUMMARY_PATH = REPORTS_DIR / "summary.md"

SEED: int = 42
TARGET: str = "diabetes"

FEATURE_PRIORITY: List[str] = [
    "heartRate",
    "BMI",
    "sysBP",
    "diaBP",
    "totChol",
    "age",
    "sex",
    "prevalentHyp",
    "BPMeds",
    "is_smoking",
    "cigsPerDay",
    "prevalentStroke",
]

EXCLUDED_FEATURES: List[str] = ["id", "glucose", "education"]

CANONICAL_MAP: Dict[str, str] = {
    **{feature.lower(): feature for feature in FEATURE_PRIORITY},
    TARGET.lower(): TARGET,
    **{feature.lower(): feature for feature in EXCLUDED_FEATURES},
}

NUMERIC_FEATURES: List[str] = [
    "heartRate",
    "BMI",
    "sysBP",
    "diaBP",
    "totChol",
    "age",
    "cigsPerDay",
]

CATEGORICAL_FEATURES: List[str] = [
    "sex",
    "prevalentHyp",
    "BPMeds",
    "is_smoking",
    "prevalentStroke",
]

ZERO_AS_NAN_FEATURES: List[str] = [
    "heartRate",
    "sysBP",
    "diaBP",
    "totChol",
    "BMI",
    "cigsPerDay",
]

# Varianti comuni dei nomi colonna -> nome canonico
FEATURE_SYNONYMS: Dict[str, str] = {
    "prevalmenthype": "prevalentHyp",
    "prevalmenthyp": "prevalentHyp",
    "prevalenthype": "prevalentHyp",
    "bpmeds": "BPMeds",
    "bpmids": "BPMeds",
    "bpmmeds": "BPMeds",
    "prevalntstrong": "prevalentStroke",
    "prevalntstroke": "prevalentStroke",
}

BINARY_TRUE_VALUES = {"1", "true", "yes", "y", "male", "m", "smoker", "positive"}
BINARY_FALSE_VALUES = {"0", "false", "no", "n", "female", "f", "non-smoker", "negative"}


def normalize_column_name(name: str) -> str:
    """Normalize a raw column name into the canonical form."""
    cleaned = name.strip().replace(" ", "")
    lowered = cleaned.lower()
    if lowered in FEATURE_SYNONYMS:
        return FEATURE_SYNONYMS[lowered]
    if lowered in CANONICAL_MAP:
        return CANONICAL_MAP[lowered]
    return cleaned


def set_global_seed(seed: int = SEED) -> None:
    """Fix Python and NumPy RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
