"""Modèles de référence (baselines).

Usage : python -m src.baseline
Produit : reports/baseline.json

Rôle de ce script
-----------------
Avant d'entraîner un modèle « sérieux », on fixe deux points de comparaison :

1. Un modèle trivial (classe majoritaire) : le plancher absolu. Tout modèle
   qui ne le bat pas nettement n'a rien appris.
2. Une régression logistique : le modèle simple et interprétable. Un modèle
   plus complexe (forêt, boosting) devra la dépasser pour justifier sa
   complexité.

Garanties contre la fuite de données
------------------------------------
- Le jeu de test est isolé AVANT toute transformation.
- Le prétraitement est dans un Pipeline : il n'est ajusté que sur
  l'entraînement, y compris à l'intérieur de chaque pli de validation croisée.
- Le nouveau mois n'est jamais utilisé ici : il représente le futur et
  servira au monitoring.
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

# Seuil de décision : une probabilité >= SEUIL est classée « panne ».
# 0,5 est la valeur par défaut, sans justification métier ; le seuil sera
# optimisé à l'étape d'entraînement (sur les prédictions hors-pli).
SEUIL = 0.5


def split(df: pd.DataFrame):
    """Sépare l'historique en entraînement (80 %) et test (20 %).

    - stratify : le taux de panne est conservé à l'identique dans les
      deux jeux, sinon le test pourrait être artificiellement facile ou difficile.
    - random_state : découpage reproductible ; chaque script (baseline, train)
      obtient exactement le même jeu de test.
    - Le test n'est utilisé qu'une fois, pour l'évaluation finale.
    """
    return train_test_split(
        df[FEATURES], df[TARGET], test_size=TEST_SIZE,
        stratify=df[TARGET], random_state=RANDOM_STATE,
    )


def preprocessor() -> ColumnTransformer:
    """Prétraitement appliqué colonne par colonne.

    - StandardScaler sur les numériques : centre-réduit chaque variable
      (moyenne 0, écart-type 1). Indispensable pour la régression logistique,
      dont la régularisation pénalise les coefficients : sans mise à l'échelle,
      une variable en centaines (usagers_jour) serait traitée différemment
      d'une variable en unités (pannes_12m).
    - OneHotEncoder sur les catégorielles (type_pompe, region) : une colonne
      0/1 par modalité. handle_unknown="ignore" évite un plantage en
      production si une modalité inconnue apparaît (nouvelle région) : elle
      est encodée comme un vecteur de zéros.

    Les moyennes, écarts-types et modalités sont appris au fit(), donc
    uniquement sur les données d'entraînement.
    """
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])


def modeles() -> dict:
    """Les deux modèles de référence, chacun encapsulé dans un Pipeline.

    Le Pipeline enchaîne prétraitement et classifieur : un seul objet à
    entraîner, sauvegarder et servir, et aucun risque d'appliquer en
    production un prétraitement différent de celui de l'entraînement.
    """
    return {
        # Prédit toujours la classe la plus fréquente (« pas de panne »).
        # Sa probabilité de panne est constante (0,36) : il ne classe aucun
        # forage avant un autre. C'est le niveau zéro de l'apprentissage.
        "dummy_majoritaire": Pipeline([
            ("prep", preprocessor()),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        # Modèle linéaire : log(p / (1 - p)) = b0 + b1*x1 + ... + bn*xn.
        # - class_weight="balanced" : poids inversement proportionnel à la
        #   fréquence de chaque classe (n / (2 * n_classe)), soit environ 1,38
        #   pour une panne et 0,79 pour une non-panne : une panne pèse ~1,75
        #   fois plus dans la fonction de perte. Compense le déséquilibre
        #   64 / 36 et évite un modèle qui sous-détecte les pannes.
        # - max_iter=1000 : laisse à l'optimiseur le temps de converger.
        # - Régularisation L2 avec C=1 (valeur par défaut) : limite la taille
        #   des coefficients pour réduire le surapprentissage.
        "regression_logistique": Pipeline([
            ("prep", preprocessor()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]),
    }


def metriques(y_true, proba, seuil=SEUIL) -> dict:
    """Calcule les métriques d'évaluation.

    Deux familles :
    - Métriques de classement (sur les probabilités, indépendantes du seuil) :
      mesurent si le modèle place les forages à risque avant les autres.
    - Métriques de décision (sur les prédictions 0/1, dépendantes du seuil) :
      mesurent ce qui se passe concrètement quand on planifie les visites.

    Notations : VP = pannes détectées, FN = pannes ratées,
                FP = visites inutiles.
    """
    pred = (proba >= seuil).astype(int)
    return {
        # PR-AUC (average precision) : aire sous la courbe précision-rappel.
        # Métrique principale de sélection. Elle se concentre sur la classe
        # panne ; le hasard vaut le taux de positifs, pas 0,5.
        "pr_auc": average_precision_score(y_true, proba),
        # ROC-AUC : probabilité qu'un forage tombé en panne reçoive un score
        # plus élevé qu'un forage sain. 0,5 = hasard, 1 = classement parfait.
        # Complémentaire, mais trop optimiste quand les négatifs dominent.
        "roc_auc": roc_auc_score(y_true, proba),
        # Rappel = VP / (VP + FN) : part des pannes réelles détectées.
        # Priorité métier : une panne ratée prive un village d'eau.
        "rappel": recall_score(y_true, pred, zero_division=0),
        # Précision = VP / (VP + FP) : part des alertes justifiées.
        # Garde-fou : trop basse, les équipes se déplacent pour rien.
        "precision": precision_score(y_true, pred, zero_division=0),
        # F2 = 5 * P * R / (4 * P + R) : moyenne harmonique qui donne au rappel
        # un poids 2 fois plus grand qu'à la précision (beta = 2).
        # Sert à choisir le seuil de décision.
        "f2": fbeta_score(y_true, pred, beta=2, zero_division=0),
    }


def main():
    df = pd.read_csv(HIST_PATH)
    X_train, X_test, y_train, y_test = split(df)

    # Validation croisée stratifiée à 5 plis sur l'entraînement uniquement :
    # chaque pli sert une fois de validation, les 4 autres d'entraînement.
    # La moyenne donne une estimation plus stable que le seul jeu de test,
    # et l'écart-type mesure la variabilité du score.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    resultats = {"taux_panne_test": float(y_test.mean())}
    for nom, modele in modeles().items():
        # 1. Estimation par validation croisée (le Pipeline est ré-ajusté
        #    dans chaque pli, donc le scaler ne voit jamais le pli de validation).
        scores_cv = cross_validate(
            modele, X_train, y_train, cv=cv,
            scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
        )
        # 2. Entraînement sur tout l'entraînement, évaluation unique sur le test.
        modele.fit(X_train, y_train)
        proba = modele.predict_proba(X_test)[:, 1]  # probabilité de la classe 1
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