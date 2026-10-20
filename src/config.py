"""Configuration partagée du projet."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"

HIST_PATH = DATA_RAW / "forages_historique.csv"
NEW_PATH = DATA_RAW / "forages_nouveau_mois.csv"

TARGET = "panne_90j"
NUM_FEATURES = [
    "age_ans",
    "profondeur_m",
    "debit_m3h",
    "usagers_jour",
    "jours_depuis_maintenance",
    "pannes_12m",
]
CAT_FEATURES = ["type_pompe", "region"]
FEATURES = NUM_FEATURES + CAT_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.2