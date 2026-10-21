"""Configuration partagée du projet.

Rassembler en un seul endroit tout ce que plusieurs scripts doivent partager :
chemins, noms de colonnes, graine aléatoire et taille du jeu de test.

Pourquoi ce fichier est important en MLOps :
- Cohérence : eda.py, baseline.py, tentrainement.py (et plus tard l'API et le
  monitoring) utilisent exactement les mêmes colonnes et le même découpage.
  Une variable ajoutée ici est prise en compte partout.
- Reproductibilité : la graine et la taille du test sont fixées une fois ;
  relancer un script redonne les mêmes résultats.
- Pas de valeurs codées en dur dispersées dans le code, source classique
  d'incohérences entre entraînement et production.
"""
from pathlib import Path

# --- Chemins -----------------------------------------------------------------
# Racine du dépôt calculée à partir de l'emplacement de ce fichier
# (src/config.py -> parents[1] = racine), et non du dossier courant :
# les scripts fonctionnent quel que soit l'endroit d'où on les lance
# (terminal, pytest, conteneur Docker, CI).
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data"   # données sources, versionnées, jamais modifiées
REPORTS = ROOT / "reports"         # rapports générés par les scripts

# Historique : sert à l'entraînement et à l'évaluation (train / test).
HIST_PATH = DATA_RAW / "forages_historique.csv"
# Nouveau mois : simule les données de production. Jamais utilisé pour
# entraîner le modèle initial ; il sert à détecter la dérive.
NEW_PATH = DATA_RAW / "forages_nouveau_mois.csv"

# --- Variables -----------------------------------------------------------------
# Cible : 1 si le forage tombe en panne dans les 90 jours suivants, 0 sinon.
TARGET = "panne_90j"

# Variables numériques : standardisées dans le pipeline (StandardScaler).
NUM_FEATURES = [
    "age_ans",                   # âge du forage (années)
    "profondeur_m",              # profondeur (mètres)
    "debit_m3h",                 # débit (m³/h)
    "usagers_jour",              # nombre d'usagers par jour (charge d'usage)
    "jours_depuis_maintenance",  # ancienneté de la dernière maintenance
    "pannes_12m",                # pannes des 12 mois précédant la prédiction
]
# Variables catégorielles : encodées en one-hot dans le pipeline.
CAT_FEATURES = ["type_pompe", "region"]

# Liste complète et ORDONNÉE des entrées du modèle. Toute colonne absente de
# cette liste n'entre jamais dans le modèle, ce qui
# protège contre l'utilisation accidentelle de la cible comme variable.
FEATURES = NUM_FEATURES + CAT_FEATURES

# --- Reproductibilité ------------------------------------------------------------
# Graine unique pour tout ce qui est aléatoire : découpage train/test,
# plis de validation croisée, forêt aléatoire, boosting.
RANDOM_STATE = 42
# Part de l'historique réservée au test final : 20 % (600 forages).
TEST_SIZE = 0.2