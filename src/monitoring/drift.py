"""Surveillance du modèle : dérive des données, performance et ré-entraînement.

Usage :
    python -m src.monitoring.drift                 # rapport de surveillance
    python -m src.monitoring.drift --reentrainer   # + ré-entraînement si la règle le demande

Produit : reports/drift.json

Principe
--------
Référence = l'historique (les données vues à l'entraînement).
Production = le nouveau mois.

1. Dérive des données : chaque variable du nouveau mois est comparée à
   l'historique (PSI, et test de Kolmogorov-Smirnov pour les numériques).
2. Dérive des prédictions : la part de forages signalés à risque change-t-elle ?
3. Performance : les vraies pannes du nouveau mois étant connues, on mesure
   le rappel et la PR-AUC du modèle en production.
4. Règle de ré-entraînement (ci-dessous). Avec --reentrainer, un nouveau
   modèle est entraîné, comparé à l'ancien, et promu seulement s'il est meilleur.
"""
import argparse
import json
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from src.baseline import metriques, split
from src.config import (
    FEATURES, HIST_PATH, NEW_PATH, NUM_FEATURES, RANDOM_STATE,
    REPORTS, TARGET,
)
from src.entrainement import (
    CANDIDATS, EXPERIMENT, MODELS_DIR, REGISTERED_NAME, TRACKING_URI,
    choisir_seuil, construire_pipeline,
)

# --- Règle de ré-entraînement ----------------------------------------------------
# PSI (Population Stability Index), seuils usuels :
#   < 0,10 stable ; 0,10 à 0,20 dérive modérée ; > 0,20 dérive forte.
PSI_FORT = 0.20
# Performance minimale en production (mêmes valeurs que le garde-fou des tests).
RAPPEL_MIN = 0.85
PR_AUC_MIN = 0.55
# Un nouveau modèle n'est promu que s'il gagne au moins ce montant de PR-AUC.
GAIN_MIN = 0.01


def psi(reference: pd.Series, courant: pd.Series, n_bins: int = 10) -> float:
    """PSI = somme (p_cour - p_ref) * ln(p_cour / p_ref) sur des intervalles.

    Numériques : intervalles = déciles de la référence, ouverts aux extrémités
    pour capter les valeurs hors de la plage d'entraînement.
    Catégorielles : une case par modalité.
    """
    if reference.dtype == object or str(reference.dtype) in ("category", "string", "str"):
        cats = sorted(set(reference) | set(courant))
        p_ref = reference.value_counts(normalize=True).reindex(cats, fill_value=0)
        p_cur = courant.value_counts(normalize=True).reindex(cats, fill_value=0)
    else:
        bords = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
        bords[0], bords[-1] = -np.inf, np.inf
        p_ref = pd.cut(reference, bords).value_counts(normalize=True, sort=False)
        p_cur = pd.cut(courant, bords).value_counts(normalize=True, sort=False)
    eps = 1e-4  # évite log(0) pour une case vide
    p_ref, p_cur = p_ref.values + eps, p_cur.values + eps
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def derive_donnees(hist: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par variable : PSI, test KS, valeurs hors plage."""
    lignes = []
    for c in FEATURES:
        ligne = {"variable": c, "psi": psi(hist[c], new[c])}
        if c in NUM_FEATURES:
            # KS : écart maximal entre les deux fonctions de répartition.
            # p-value < 0,01 : les distributions diffèrent significativement.
            ks = ks_2samp(hist[c], new[c])
            ligne |= {"ks_stat": ks.statistic, "ks_pvalue": ks.pvalue,
                      "hors_plage_%": 100 * ((new[c] > hist[c].max())
                                              | (new[c] < hist[c].min())).mean()}
        ligne["niveau"] = ("forte" if ligne["psi"] > PSI_FORT
                           else "moderee" if ligne["psi"] > 0.10 else "stable")
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def evaluer_production(modele, seuil, hist, new) -> dict:
    """Dérive des prédictions et performance réelle sur le nouveau mois."""
    _, X_test, _, y_test = split(hist)  # même jeu de test qu'à l'entraînement
    p_ref = modele.predict_proba(X_test)[:, 1]
    p_new = modele.predict_proba(new[FEATURES])[:, 1]
    return {
        "taux_alerte_reference": float((p_ref >= seuil).mean()),
        "taux_alerte_nouveau_mois": float((p_new >= seuil).mean()),
        "proba_moyenne_reference": float(p_ref.mean()),
        "proba_moyenne_nouveau_mois": float(p_new.mean()),
        "taux_panne_reel_nouveau_mois": float(new[TARGET].mean()),
        "reference": metriques(y_test, p_ref, seuil),
        "nouveau_mois": metriques(new[TARGET], p_new, seuil),
    }


def decision(derive: pd.DataFrame, perf: dict) -> dict:
    """Règle : ré-entraîner si dérive forte sur au moins une variable,
    ou si la performance réelle passe sous le minimum."""
    m = perf["nouveau_mois"]
    raisons = [f"dérive forte sur {v} (PSI {p:.2f})"
               for v, p in derive.loc[derive.niveau == "forte", ["variable", "psi"]].values]
    if m["rappel"] < RAPPEL_MIN:
        raisons.append(f"rappel {m['rappel']:.2f} < {RAPPEL_MIN}")
    if m["pr_auc"] < PR_AUC_MIN:
        raisons.append(f"PR-AUC {m['pr_auc']:.2f} < {PR_AUC_MIN}")
    return {"reentrainer": bool(raisons), "raisons": raisons}


def reentrainer(ancien, metadata, hist, new) -> dict:
    """Ré-entraîne sur historique + moitié du nouveau mois, compare sur l'autre moitié.

    - L'autre moitié du nouveau mois n'est jamais vue : comparaison honnête
      entre ancien et nouveau modèle sur les données récentes.
    - Le jeu de test historique reste exclu : il sert de contrôle de
      non-régression (le nouveau modèle doit rester bon sur l'ancien contexte).
    - Même type de modèle et même méthode de seuil qu'à l'entraînement initial.
    """
    new_train, new_eval = train_test_split(
        new, test_size=0.5, stratify=new[TARGET], random_state=RANDOM_STATE)
    X_hist_train, X_hist_test, y_hist_train, y_hist_test = split(hist)
    X = pd.concat([X_hist_train, new_train[FEATURES]])
    y = pd.concat([y_hist_train, new_train[TARGET]])

    nom = metadata["modele"]
    pipe = construire_pipeline(CANDIDATS[nom])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    seuil = choisir_seuil(y, cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1])
    pipe.fit(X, y)

    def scores(modele, s):
        return {"nouveau_mois_eval": metriques(new_eval[TARGET],
                                               modele.predict_proba(new_eval[FEATURES])[:, 1], s),
                "test_historique": metriques(y_hist_test,
                                             modele.predict_proba(X_hist_test)[:, 1], s)}

    avant, apres = scores(ancien, metadata["seuil"]), scores(pipe, seuil)
    gain = apres["nouveau_mois_eval"]["pr_auc"] - avant["nouveau_mois_eval"]["pr_auc"]
    promu = (gain >= GAIN_MIN
             and apres["nouveau_mois_eval"]["rappel"] >= RAPPEL_MIN
             and apres["test_historique"]["pr_auc"] >= PR_AUC_MIN)

    # Traçage MLflow du ré-entraînement, promu ou non.
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"reentrainement_{nom}") as run:
        mlflow.log_params({"modele": nom, "seuil": seuil, "donnees": "historique+nouveau_mois",
                           "promu": promu})
        mlflow.log_metrics({f"{jeu}_{k}": v for jeu, ms in apres.items() for k, v in ms.items()})
        mlflow.log_metric("gain_pr_auc", gain)
        mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")

    if promu:
        version = mlflow.register_model(f"runs:/{run.info.run_id}/model", REGISTERED_NAME)
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(REGISTERED_NAME, "production", version.version)
        client.set_model_version_tag(REGISTERED_NAME, version.version, "seuil", str(seuil))
        joblib.dump(pipe, MODELS_DIR / "model.joblib")
        metadata |= {"seuil": seuil, "mlflow_run_id": run.info.run_id,
                     "registre_version": version.version, "test": apres["test_historique"],
                     "reentraine_le": datetime.now(timezone.utc).isoformat()}
        (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {"seuil_nouveau": seuil, "avant": avant, "apres": apres,
            "gain_pr_auc": gain, "promu": promu}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reentrainer", action="store_true",
                        help="ré-entraîner si la règle le demande")
    args = parser.parse_args()

    hist, new = pd.read_csv(HIST_PATH), pd.read_csv(NEW_PATH)
    modele = joblib.load(MODELS_DIR / "model.joblib")
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))

    derive = derive_donnees(hist, new)
    perf = evaluer_production(modele, metadata["seuil"], hist, new)
    dec = decision(derive, perf)
    rapport = {"date": datetime.now(timezone.utc).isoformat(),
               "modele": metadata["modele"], "seuil": metadata["seuil"],
               "derive_donnees": derive.round(4).to_dict(orient="records"),
               "production": perf, "decision": dec}

    print(derive.round(3).to_string(index=False))
    for jeu in ("reference", "nouveau_mois"):
        m = perf[jeu]
        print(f"\n{jeu:13s} PR-AUC {m['pr_auc']:.3f}  rappel {m['rappel']:.3f}  "
              f"précision {m['precision']:.3f}")
    print(f"\nTaux d'alerte : {perf['taux_alerte_reference']:.2f} -> "
          f"{perf['taux_alerte_nouveau_mois']:.2f}")
    print("Ré-entraîner :", dec["reentrainer"], dec["raisons"])

    if args.reentrainer and dec["reentrainer"]:
        res = reentrainer(modele, metadata, hist, new)
        rapport["reentrainement"] = res
        for jeu in ("nouveau_mois_eval", "test_historique"):
            print(f"\n{jeu}: PR-AUC {res['avant'][jeu]['pr_auc']:.3f} -> "
                  f"{res['apres'][jeu]['pr_auc']:.3f} ; rappel "
                  f"{res['avant'][jeu]['rappel']:.3f} -> {res['apres'][jeu]['rappel']:.3f}")
        print(f"Gain PR-AUC : {res['gain_pr_auc']:+.3f} -> "
              f"{'PROMU (alias production)' if res['promu'] else 'non promu, ancien modèle conservé'}")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "drift.json").write_text(json.dumps(rapport, indent=2, default=float),
                                        encoding="utf-8")


if __name__ == "__main__":
    main()
