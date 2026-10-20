"""Modèles de référence.

Usage : python -m src.baseline
Produit : reports/baseline.json

- Le jeu de test est isolé AVANT toute transformation (pas de fuite).
- Le prétraitement est dans un Pipeline : il n'est ajusté que sur l'entraînement,
  y compris à l'intérieur de chaque pli de validation croisée.
- Le nouveau mois n'est jamais utilisé ici : il sert au monitoring.
"""
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, fbeta_score, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CAT_FEATURES, FEATURES, HIST_PATH, NUM_FEATURES, RANDOM_STATE, REPORTS,
    TARGET, TEST_SIZE,
)

SEUIL = 0.5  # seuil par défaut ; il sera optimisé à l'étape d'entraînement


def split(df: pd.DataFrame):
    return train_test_split(
        df[FEATURES], df[TARGET], test_size=TEST_SIZE,
        stratify=df[TARGET], random_state=RANDOM_STATE,
    )


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])


def modeles() -> dict:
    return {
        "dummy_majoritaire": Pipeline([
            ("prep", preprocessor()),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        "regression_logistique": Pipeline([
            ("prep", preprocessor()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]),
    }


def metriques(y_true, proba, seuil=SEUIL) -> dict:
    pred = (proba >= seuil).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "rappel": recall_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "f2": fbeta_score(y_true, pred, beta=2, zero_division=0),
    }


def main():
    df = pd.read_csv(HIST_PATH)
    X_train, X_test, y_train, y_test = split(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    resultats = {"taux_panne_test": float(y_test.mean())}
    for nom, modele in modeles().items():
        scores_cv = cross_validate(
            modele, X_train, y_train, cv=cv,
            scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
        )
        modele.fit(X_train, y_train)
        proba = modele.predict_proba(X_test)[:, 1]
        resultats[nom] = {
            "cv_pr_auc_moy": scores_cv["test_pr_auc"].mean(),
            "cv_pr_auc_std": scores_cv["test_pr_auc"].std(),
            "test": metriques(y_test, proba),
        }

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "baseline.json"
    out.write_text(json.dumps(resultats, indent=2, default=float), encoding="utf-8")
    print(json.dumps(resultats, indent=2, default=float))
    print(f"\nRésultats écrits : {out}")


if __name__ == "__main__":
    main()