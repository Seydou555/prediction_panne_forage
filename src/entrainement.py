"""Entraînement tracé dans MLflow, sélection et sauvegarde du modèle.

Usage : python -m src.entrainement
Produit :
- un run MLflow par configuration candidate (expérience "pannes-forages")
- le meilleur modèle enregistré dans le registre MLflow ("pannes-forages-model")
- models/model.joblib et models/metadata.json (utilisés par l'API)
- reports/selection.json (tableau de comparaison et justification)

Déroulement
-------
Pour chaque modèle candidat :
  1. validation croisée sur l'entraînement (PR-AUC, ROC-AUC) ;
  2. choix du seuil de décision sur les prédictions hors-pli ;
  3. entraînement final, puis évaluation unique sur le jeu de test ;
  4. enregistrement de tout dans un run MLflow (paramètres, métriques, modèle).
Ensuite : sélection selon une règle écrite à l'avance, enregistrement du
modèle choisi dans le registre, copie locale pour l'API.

Le découpage, le prétraitement et les métriques sont importés de
src/baseline.py : les candidats sont évalués exactement comme le modèle de
référence, sur le même jeu de test, ce qui rend la comparaison valable.
"""
import json
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from src.baseline import metriques, preprocessor, split
from src.config import FEATURES, HIST_PATH, RANDOM_STATE, REPORTS, ROOT

# Suivi MLflow dans une base SQLite locale : le registre de modèles exige un
# backend de type base de données (un simple dossier mlruns/ ne suffit pas).
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
EXPERIMENT = "pannes-forages"            # regroupe tous les runs du projet
REGISTERED_NAME = "pannes-forages-model"  # nom du modèle dans le registre
MODELS_DIR = ROOT / "models"

# Paramètres de décision, fixés AVANT de voir les résultats :
# - PRECISION_MIN : au moins 40 % des alertes doivent être de vraies pannes,
#   sinon les équipes perdent confiance. Paramètre métier, à valider avec le
#   service de l'hydraulique (le monter si les équipes sont peu nombreuses).
# - TOLERANCE_PR_AUC : un écart de PR-AUC inférieur à 0,01 est jugé non
#   significatif (bien en dessous de l'écart-type de la CV, environ 0,03).
PRECISION_MIN = 0.40
TOLERANCE_PR_AUC = 0.01

# Candidats, du plus simple au plus complexe.
# Tous utilisent class_weight="balanced" pour compenser le déséquilibre 64/36.
CANDIDATS = {
    # Régression logistique, régularisation L2 standard (C = 1).
    "logreg_C1": LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000),
    # Même modèle, régularisation 10 fois plus forte (C petit = coefficients
    # plus contraints) : teste si le modèle de base surapprend.
    "logreg_C0.1": LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000),
    # Forêt aléatoire : moyenne de 300 arbres entraînés sur des échantillons
    # différents. Capte les interactions et effets non linéaires.
    # max_depth=8 et min_samples_leaf=10 limitent le surapprentissage.
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=10,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    ),
    # Gradient boosting : arbres ajoutés un par un, chacun corrigeant les
    # erreurs des précédents. Souvent le plus performant sur données
    # tabulaires. learning_rate faible + l2_regularization + max_depth=4
    # pour rester prudent sur 2 400 lignes d'entraînement.
    "hist_gradient_boosting": HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_depth=4, l2_regularization=1.0,
        class_weight="balanced", random_state=RANDOM_STATE,
    ),
}

def construire_pipeline(clf) -> Pipeline:
    """Prétraitement + classifieur dans un seul objet.

    clone() crée une copie neuve, non entraînée, du classifieur : chaque run
    part de zéro et le dictionnaire CANDIDATS n'est jamais modifié.
    """
    return Pipeline([("prep", preprocessor()), ("clf", clone(clf))])

def choisir_seuil(y_true, proba_oof) -> float:
    """Choisit le seuil de décision (probabilité au-delà de laquelle on
    planifie une visite).

    Méthode : on teste tous les seuils de 0,10 à 0,90 par pas de 0,01 ; on
    écarte ceux dont la précision est sous PRECISION_MIN ; parmi les autres,
    on garde celui qui maximise F2 (rappel deux fois plus important que la
    précision).

    Les probabilités utilisées sont « hors-pli » (out-of-fold) : chaque forage
    de l'entraînement est prédit par un modèle qui ne l'a pas vu. Le seuil est
    donc réglé sans jamais regarder le jeu de test, qui reste une évaluation
    honnête. Si aucun seuil ne respecte la contrainte, on garde 0,5.
    """
    meilleur, meilleur_f2 = 0.5, -1.0
    for s in np.arange(0.10, 0.91, 0.01):
        pred = (proba_oof >= s).astype(int)
        if precision_score(y_true, pred, zero_division=0) < PRECISION_MIN:
            continue
        f2 = fbeta_score(y_true, pred, beta=2, zero_division=0)
        if f2 > meilleur_f2:
            meilleur, meilleur_f2 = float(round(s, 2)), f2
    return meilleur


def evaluer(nom, clf, X_train, y_train, X_test, y_test, cv) -> dict:
    """Évalue un candidat et trace tout dans un run MLflow."""
    pipe = construire_pipeline(clf)
    with mlflow.start_run(run_name=nom) as run:
        # 1. Validation croisée sur l'entraînement : PR-AUC (métrique de
        #    sélection) et ROC-AUC, moyenne et écart-type sur 5 plis.
        cv_scores = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
        )
        # Probabilités hors-pli : une prédiction par forage d'entraînement,
        # faite par le modèle du pli où ce forage servait de validation.
        proba_oof = cross_val_predict(pipe, X_train, y_train, cv=cv,
                                      method="predict_proba")[:, 1]
        seuil = choisir_seuil(y_train, proba_oof)
        # Rappel, précision et F2 attendus avec ce seuil (estimation CV).
        cv_seuil = metriques(y_train, proba_oof, seuil)

        # 2. Entraînement final sur tout l'entraînement, puis évaluation
        #    unique sur le test avec le seuil déjà choisi.
        pipe.fit(X_train, y_train)
        proba_test = pipe.predict_proba(X_test)[:, 1]
        test = metriques(y_test, proba_test, seuil)

        # 3. Traçage MLflow
        # Paramètres : tout ce qu'il faut pour reproduire le run, y compris
        # tous les hyperparamètres du classifieur (préfixe clf__).
        mlflow.log_params({"modele": nom, "seuil": seuil,
                           "precision_min": PRECISION_MIN,
                           **{f"clf__{k}": v for k, v in clf.get_params().items()}})
        # Métriques : préfixe cv_ (estimation sur l'entraînement) ou test_
        # (évaluation finale), pour les comparer côte à côte dans l'interface.
        mlflow.log_metrics({
            "cv_pr_auc": cv_scores["test_pr_auc"].mean(),
            "cv_pr_auc_std": cv_scores["test_pr_auc"].std(),
            "cv_roc_auc": cv_scores["test_roc_auc"].mean(),
            **{f"cv_{k}": v for k, v in cv_seuil.items() if k not in ("pr_auc", "roc_auc")},
            **{f"test_{k}": v for k, v in test.items()},
        })
        # Signature : schéma des entrées (noms et types des colonnes) et de la
        # sortie. MLflow s'en sert pour refuser une entrée mal formée.
        # input_example : exemple de requête stocké avec le modèle.
        # cloudpickle : le format par défaut (skops) refuse les modèles à base
        # d'arbres sans déclaration explicite des types ; on produit nous-mêmes
        # ces modèles, cloudpickle est donc acceptable.
        signature = infer_signature(X_train, pipe.predict_proba(X_train)[:, 1])
        mlflow.sklearn.log_model(pipe, name="model", signature=signature,
                                 input_example=X_train.head(3),
                                 serialization_format="cloudpickle")

        return {"nom": nom, "run_id": run.info.run_id, "pipeline": pipe,
                "seuil": seuil, "cv_pr_auc": cv_scores["test_pr_auc"].mean(),
                "cv_pr_auc_std": cv_scores["test_pr_auc"].std(),
                "cv_f2": cv_seuil["f2"], "test": test}


def selectionner(resultats: list) -> dict:
    """Règle de sélection, écrite avant de voir les résultats.

    1. On retient la meilleure PR-AUC en validation croisée (pas sur le test :
       choisir sur le test reviendrait à l'utiliser pour régler le modèle).
    2. Tout candidat à moins de TOLERANCE_PR_AUC du meilleur est jugé
       équivalent ; on prend alors le premier de la liste, donc le plus
       simple (principe de parcimonie).
    """
    best = max(r["cv_pr_auc"] for r in resultats)
    return next(r for r in resultats if r["cv_pr_auc"] >= best - TOLERANCE_PR_AUC)


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)  # crée l'expérience si elle n'existe pas

    df = pd.read_csv(HIST_PATH)
    X_train, X_test, y_train, y_test = split(df)  # même découpage que baseline
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    resultats = [evaluer(n, c, X_train, y_train, X_test, y_test, cv)
                 for n, c in CANDIDATS.items()]
    choix = selectionner(resultats)

    # Registre MLflow : chaque exécution crée une nouvelle version du modèle.
    # L'alias "production" désigne la version à servir ; le déplacer suffit
    # pour promouvoir un nouveau modèle ou revenir en arrière.
    # Le seuil est attaché à la version : modèle et seuil vont ensemble.
    version = mlflow.register_model(f"runs:/{choix['run_id']}/model", REGISTERED_NAME)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(REGISTERED_NAME, "production", version.version)
    client.set_model_version_tag(REGISTERED_NAME, version.version, "seuil", str(choix["seuil"]))

    # Copie locale pour l'API : l'image Docker charge ce fichier sans dépendre
    # d'un serveur MLflow. metadata.json garde le lien avec le run (traçabilité)
    # et fournit le seuil et la liste des variables attendues.
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(choix["pipeline"], MODELS_DIR / "model.joblib")
    metadata = {
        "modele": choix["nom"], "seuil": choix["seuil"], "features": FEATURES,
        "mlflow_run_id": choix["run_id"], "registre_version": version.version,
        "cv_pr_auc": choix["cv_pr_auc"], "test": choix["test"],
        "entraine_le": datetime.now(timezone.utc).isoformat(),
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Rapport de sélection versionné dans Git : trace du choix et de sa règle.
    tableau = pd.DataFrame([{
        "modele": r["nom"], "cv_pr_auc": r["cv_pr_auc"], "cv_pr_auc_std": r["cv_pr_auc_std"],
        "seuil": r["seuil"], "cv_f2": r["cv_f2"], "test_pr_auc": r["test"]["pr_auc"],
        "test_rappel": r["test"]["rappel"], "test_precision": r["test"]["precision"],
        "test_f2": r["test"]["f2"],
    } for r in resultats]).round(3)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "selection.json").write_text(json.dumps({
        "critere": f"PR-AUC CV max ; à moins de {TOLERANCE_PR_AUC} près, modèle le plus simple",
        "choix": choix["nom"], "seuil": choix["seuil"],
        "comparaison": tableau.to_dict(orient="records"),
    }, indent=2), encoding="utf-8")

    print(tableau.to_string(index=False))
    print(f"\nModèle retenu : {choix['nom']} (seuil {choix['seuil']}), "
          f"version {version.version} du registre, alias 'production'")


if __name__ == "__main__":
    main()