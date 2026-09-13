"""Central configuration for HealthBandhu V.

Every constant that the notebook and the app must agree on lives here, so a
change (for example an emergency weight) is made once and applies everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
APP_TITLE = "HealthBandhu V"
APP_TAGLINE = "AI-assisted multiclass disease screening and clinical decision support"
AUTHOR = "Joy Ghatak"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("HEALTHBANDHU_MODELS_DIR", PROJECT_ROOT / "models"))
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATASET_FILENAME = "medical_dataset.csv"
DATASET_PATH = DATA_DIR / DATASET_FILENAME
DATASET_GDRIVE_FILE_ID = "1-vtq4LIJI2JdKM6dWFhdUM9ub8Zcll10"
TARGET_COLUMN = "diseases"

# ---------------------------------------------------------------------------
# Model artifacts (written by the notebook, read by the app)
# ---------------------------------------------------------------------------
DNN_MODEL_FILE = "healthbandhu_dnn.keras"
NB_PARAMS_FILE = "healthbandhu_nb_params.npz"
SYMPTOMS_FILE = "healthbandhu_symptoms.json"
DISEASES_FILE = "healthbandhu_diseases.json"
PROFILES_FILE = "healthbandhu_disease_profiles.npy"
METADATA_FILE = "model_metadata.json"

# Older notebook versions saved the name lists with joblib; still accepted.
LEGACY_SYMPTOMS_FILE = "healthbandhu_symptoms.pkl"
LEGACY_DISEASES_FILE = "healthbandhu_diseases.pkl"

# ---------------------------------------------------------------------------
# Training defaults (used by the notebook)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
MIN_SAMPLES_PER_CLASS = 2   # classes below this cannot be stratified -> removed
TEST_SIZE = 0.20            # fraction of all rows held out for final testing
VAL_SIZE = 0.10             # fraction of the remaining rows used for tuning

# ---------------------------------------------------------------------------
# Phase 3: emergency detection layer
# ---------------------------------------------------------------------------
# The first eight entries are the original HealthBandhu weights. The rest are
# recognised red-flag symptoms that exist in the dataset (stroke signs,
# gastrointestinal bleeding, airway compromise). Edit weights here only.
EMERGENCY_SYMPTOMS = {
    "shortness of breath": 5,
    "sharp chest pain": 5,
    "irregular heartbeat": 5,
    "breathing fast": 4,
    "chest tightness": 4,
    "throat swelling": 4,
    "fainting": 5,
    "seizures": 5,
    "difficulty breathing": 5,
    "vomiting blood": 5,
    "slurring words": 5,
    "focal weakness": 4,
    "hemoptysis": 4,
    "melena": 4,
    "swollen tongue": 4,
}

# (level, minimum score) checked from the top down.
EMERGENCY_LEVELS = (
    ("CRITICAL", 10),
    ("HIGH RISK", 5),
    ("MODERATE RISK", 2),
)
LOW_RISK = "LOW RISK"

EMERGENCY_ACTIONS = {
    "CRITICAL": "Seek emergency care now. Go to the nearest emergency department or call an ambulance.",
    "HIGH RISK": "Get medical attention today, at an emergency department or urgent care clinic, "
                 "especially if the symptoms are getting worse.",
    "MODERATE RISK": "See a doctor within 24 to 48 hours and watch closely for any worsening.",
    "LOW RISK": "No red-flag symptoms were reported. Consult a doctor if symptoms persist or worsen.",
}
EMERGENCY_CONTACT_NOTE = "In India, dial 112 for emergencies or 108 for an ambulance."

# ---------------------------------------------------------------------------
# Phase 10: confidence layer
# ---------------------------------------------------------------------------
CONFIDENCE_LEVELS = (
    ("VERY HIGH", 0.90),
    ("HIGH", 0.75),
    ("MODERATE", 0.50),
)
LOW_CONFIDENCE = "LOW"
AMBIGUITY_MARGIN = 0.15            # top-1 minus top-2 probability below this = ambiguous
MIN_SYMPTOMS_FOR_RELIABLE_RESULT = 3
RARE_CLASS_TRAIN_COUNT = 20        # fewer training examples than this = rare condition

# ---------------------------------------------------------------------------
# Phase 4: rule-based clinical layer
# ---------------------------------------------------------------------------
KEY_SYMPTOM_THRESHOLD = 0.25   # symptom present in >= 25% of a disease's patients
KEY_SYMPTOMS_MIN = 3
KEY_SYMPTOMS_MAX = 12
RULE_F1_WEIGHT = 0.7           # rule score = 0.7 * F1(key symptoms) + 0.3 * coverage

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
TOP_K = 5
DEFAULT_DNN_WEIGHT = 0.8        # used only if model_metadata.json has no tuned weight
TYPICAL_SYMPTOM_MIN_PROB = 0.30  # for "commonly seen but not reported" hints

DISCLAIMER = (
    "HealthBandhu is an educational and research prototype. It does not provide a "
    "medical diagnosis and must not replace assessment by a qualified healthcare "
    "professional. Do not start, stop or change any treatment based on this result."
)
