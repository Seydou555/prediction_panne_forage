"""Tests du projet : données, pipeline, garde-fou de performance, API.

Usage (depuis la racine du dépôt) : python -m pytest -v

Les tests du modèle et de l'API ont besoin de models/model.joblib : lancer
d'abord `python -m src.entrainement`. S'il est absent, ces tests sont
ignorés (skip) au lieu d'échouer.
"""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from src.baseline import metriques, preprocessor, split
from src.config import (
    FEATURES, HIST_PATH, NEW_PATH, NUM_FEATURES, ROOT, TARGET,
)
from src.entrainement import PRECISION_MIN, choisir_seuil, selectionner

MODEL_PATH = ROOT / "models" / "model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
modele_requis = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="modèle absent : lancer python -m src.entrainement"
)

# Garde-fou de performance, sur le jeu de test et avec le seuil sauvegardé.
# Valeurs mesurées au moment du choix du modèle : PR-AUC 0,61, rappel 0,97,
# précision 0,42. Les minima laissent une marge pour les petites variations,
# mais bloquent tout modèle nettement dégradé.
PR_AUC_MIN = 0.55      # le hasard vaut 0,36
RAPPEL_MIN = 0.85      # priorité métier : ne pas rater les pannes
PRECISION_GARDE = 0.35  # limiter les visites inutiles

FORAGE_VALIDE = {
    "age_ans": 25, "profondeur_m": 32, "debit_m3h": 5.6, "usagers_jour": 730,
    "jours_depuis_maintenance": 526, "pannes_12m": 0,
    "type_pompe": "manuelle", "region": "Sud",
}


@pytest.fixture(scope="module")
def hist():
    return pd.read_csv(HIST_PATH)


# --- 1. Données ---------------------------------------------------------------

@pytest.mark.parametrize("chemin", [HIST_PATH, NEW_PATH])
def test_schema_des_donnees(chemin):
    """Les deux fichiers contiennent toutes les colonnes attendues, sans vide."""
    df = pd.read_csv(chemin)
    assert set(FEATURES + [TARGET]) <= set(df.columns)
    assert not df[FEATURES + [TARGET]].isna().any().any()


def test_cible_binaire(hist):
    assert set(hist[TARGET].unique()) == {0, 1}


def test_plages_plausibles(hist):
    """Pas de valeur négative dans les variables numériques."""
    assert (hist[NUM_FEATURES] >= 0).all().all()


# --- 2. Découpage et pipeline -------------------------------------------------

def test_split_stratifie_et_reproductible(hist):
    X_tr, X_te, y_tr, y_te = split(hist)
    assert len(X_te) == pytest.approx(0.2 * len(hist), abs=1)
    # Stratification : même taux de panne dans les deux jeux
    assert y_tr.mean() == pytest.approx(y_te.mean(), abs=0.01)
    # Reproductibilité : même graine, même jeu de test
    assert split(hist)[1].index.equals(X_te.index)
    # Pas de forage à la fois en entraînement et en test
    assert X_tr.index.intersection(X_te.index).empty


def test_preprocesseur_modalite_inconnue(hist):
    """Une région jamais vue ne doit pas faire planter le prétraitement."""
    prep = preprocessor().fit(hist[FEATURES])
    nouveau = pd.DataFrame([{**FORAGE_VALIDE, "region": "Ouest"}])
    assert prep.transform(nouveau).shape[0] == 1


def test_metriques_cas_parfait():
    y = np.array([0, 0, 1, 1])
    m = metriques(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["pr_auc"] == m["rappel"] == m["precision"] == m["f2"] == 1.0


def test_seuil_respecte_la_precision_minimale():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    proba = np.clip(y * 0.4 + rng.random(500) * 0.6, 0, 1)
    seuil = choisir_seuil(y, proba)
    pred = (proba >= seuil).astype(int)
    assert (y[pred == 1].mean()) >= PRECISION_MIN


def test_selection_privilegie_le_modele_simple():
    """À moins de 0,01 de PR-AUC, le premier candidat (le plus simple) gagne."""
    resultats = [{"nom": "simple", "cv_pr_auc": 0.650},
                 {"nom": "complexe", "cv_pr_auc": 0.655}]
    assert selectionner(resultats)["nom"] == "simple"
    resultats[1]["cv_pr_auc"] = 0.700
    assert selectionner(resultats)["nom"] == "complexe"


# --- 3. Garde-fou de performance ------------------------------------------------

@modele_requis
def test_garde_fou_performance(hist):
    """Bloque la CI si le modèle sauvegardé se dégrade sur le jeu de test."""
    import joblib
    modele = joblib.load(MODEL_PATH)
    seuil = json.loads(METADATA_PATH.read_text(encoding="utf-8"))["seuil"]
    _, X_te, _, y_te = split(hist)
    proba = modele.predict_proba(X_te)[:, 1]
    m = metriques(y_te, proba, seuil)
    assert average_precision_score(y_te, proba) >= PR_AUC_MIN, m
    assert m["rappel"] >= RAPPEL_MIN, m
    assert m["precision"] >= PRECISION_GARDE, m


# --- 4. API ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.api.main import app
    with TestClient(app) as c:  # le "with" déclenche le chargement du modèle
        yield c


@modele_requis
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


@modele_requis
def test_predict_valide(client):
    r = client.post("/predict", json=FORAGE_VALIDE)
    assert r.status_code == 200
    corps = r.json()
    assert 0 <= corps["probabilite_panne"] <= 1
    assert corps["panne_probable"] == (corps["probabilite_panne"] >= corps["seuil_utilise"])


@modele_requis
@pytest.mark.parametrize("modif", [
    {"age_ans": None},           # champ manquant
    {"age_ans": "vingt"},        # mauvais type
    {"type_pompe": None},        # catégorielle manquante
])
def test_predict_entree_invalide(client, modif):
    """Pydantic doit rejeter les entrées mal formées (422), sans appeler le modèle."""
    forage = {k: v for k, v in {**FORAGE_VALIDE, **modif}.items() if v is not None}
    assert client.post("/predict", json=forage).status_code == 422
