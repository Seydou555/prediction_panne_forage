# FastAPI : /predict, /health


import sys
from pathlib import Path

# Ajout prioritaire de la racine du projet dans sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from src.config import ROOT
from src.api.schemas import ForageInput  # Import du schéma Pydantic externe

app = FastAPI(
    title="API Prédiction Panne Forage",
    description="API permettant de prédire le risque de panne sur les forages avec validation Pydantic.",
    version="1.0.0"
)

# Chemins vers le modèle et les métadonnées
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

# Variables globales
model = None
metadata = None

@app.on_event("startup")
def load_artifacts():
    """Charge le modèle et les métadonnées au démarrage de l'API."""
    global model, metadata
    if MODEL_PATH.exists() and METADATA_PATH.exists():
        model = joblib.load(MODEL_PATH)
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        print("Modèle et métadonnées chargés avec succès !")
    else:
        model = None
        metadata = {"seuil": 0.5, "features": []}
        print("Attention : Modèle ou métadonnées introuvables.")

# 1. Endpoint de santé (Health Check)
@app.get("/health")
def health_check():
    """Vérifie si l'API tourne et si le modèle est bien chargé."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "modele_actif": metadata.get("modele", "inconnu"),
        "seuil": metadata.get("seuil", 0.5)
    }

# Route racine d'accueil
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de prédiction. Rendez-vous sur /docs pour la documentation."}

# 2. Endpoint de prédiction (/predict) avec validation Pydantic
@app.post("/predict")
def predict(data: ForageInput):
    """Reçoit les caractéristiques validées par Pydantic, 
    effectue la prédiction de probabilité et applique le seuil métier.
    """
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Le modèle n'est pas chargé. Veuillez exécuter le script d'entraînement."
        )
    
    try:
        # Conversion des données validées par Pydantic en dictionnaire puis DataFrame
        input_data = data.dict() if hasattr(data, "dict") else dict(data)
        input_df = pd.DataFrame([input_data])
        
        # Sélection stricte des colonnes dans le bon ordre défini par l'entraînement
        features_attendues = metadata.get("features", [])
        input_df = input_df[features_attendues]

        # Calcul de la probabilité de panne (classe 1)
        proba = float(model.predict_proba(input_df)[0, 1])
        
        # Application du seuil métier sauvegardé lors de l'entraînement
        seuil = metadata.get("seuil", 0.5)
        panne_probable = bool(proba >= seuil)

        return {
            "probabilite_panne": round(proba, 4),
            "seuil_utilise": seuil,
            "panne_probable": panne_probable,
            "message": "Risque de panne élevé - Intervention recommandée" if panne_probable else "Fonctionnement normal"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {str(e)}")