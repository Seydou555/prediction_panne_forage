"""Exploration des données et audit de fuite.

Usage : python -m src.eda
Produit : reports/eda.md

Rôle de ce script
-----------------
Comprendre les données AVANT de modéliser, et laisser une trace reproductible
de chaque constat (règle du projet : aucun résultat sans le script qui le
produit). Il répond à quatre questions :

1. Qualité : les données sont-elles complètes et cohérentes ?
2. Signal : quelles variables semblent liées au risque de panne ?
3. Dérive : le nouveau mois ressemble-t-il à l'historique ?
4. Fuite : une variable contient-elle une information qu'on n'aurait pas
   au moment de prédire ?

L'exploration porte sur tout l'historique : elle sert à comprendre, pas à
régler le modèle. Aucun paramètre du modèle n'est tiré d'ici, donc le jeu
de test n'est pas « contaminé ».
"""
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    CAT_FEATURES, FEATURES, HIST_PATH, NEW_PATH, NUM_FEATURES,
    RANDOM_STATE, REPORTS, TARGET,
)


def load():
    """Charge l'historique (3 000 forages, base d'apprentissage) et le
    nouveau mois (600 forages, simulation de la production)."""
    return pd.read_csv(HIST_PATH), pd.read_csv(NEW_PATH)


def qualite(df: pd.DataFrame) -> pd.DataFrame:
    """Contrôle de qualité, une ligne par colonne.

    - type : vérifie que chaque colonne est lue dans le bon type (un nombre
      lu comme texte signalerait des valeurs parasites, ex. « N/A »).
    - manquants : nombre de valeurs vides ; décide s'il faut une imputation
      dans le pipeline.
    - uniques : nombre de valeurs distinctes ; repère les colonnes constantes
      (inutiles) et confirme que la cible est bien binaire (2 valeurs).
    """
    return pd.DataFrame({
        "type": df.dtypes.astype(str),
        "manquants": df.isna().sum(),
        "uniques": df.nunique(),
    })


def comparaison(hist: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Compare l'historique et le nouveau mois, variable par variable.

    Premier signal, simple, de dérive des données :
    - moy_hist / moy_new : déplacement du centre de la distribution.
    - max_hist / max_new : un max_new > max_hist signifie que le modèle
      recevra des valeurs jamais vues à l'entraînement (extrapolation).
    - ecart_moy_% : écart relatif des moyennes, pour comparer des variables
      d'échelles différentes.
    La cible est incluse : un changement du taux de panne indique que le
    contexte lui-même a évolué.
    Les tests statistiques formels (PSI, Kolmogorov-Smirnov) viendront à
    l'étape de monitoring.
    """
    rows = []
    for c in NUM_FEATURES + [TARGET]:
        rows.append({
            "variable": c,
            "moy_hist": hist[c].mean(), "moy_new": new[c].mean(),
            "max_hist": hist[c].max(), "max_new": new[c].max(),
            "ecart_moy_%": 100 * (new[c].mean() - hist[c].mean()) / hist[c].mean(),
        })
    return pd.DataFrame(rows).round(2)


def taux_par_modalite(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Taux de panne et effectif pour chaque valeur d'une variable.

    - taux_panne : moyenne de la cible 0/1, donc la proportion de forages
      tombés en panne dans ce groupe. À comparer au taux global (0,36).
    - effectif : nombre de forages du groupe. Indispensable pour juger de la
      fiabilité du taux : un taux calculé sur 1 ou 18 forages est fragile.
    """
    return df.groupby(col)[TARGET].agg(taux_panne="mean", effectif="size").round(3)


def audit_fuite(hist: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Recherche de signaux de fuite de données.

    1. Doublons dans l'historique : une même ligne dans l'entraînement et le
       test gonflerait artificiellement le score.
    2. Recouvrement historique / nouveau mois : si des forages identiques
       apparaissaient dans les deux fichiers, le monitoring évaluerait le
       modèle sur des données qu'il a déjà vues.
    3. Pouvoir prédictif de chaque variable seule : on entraîne un petit arbre
       (profondeur 3) sur une seule variable et on mesure sa ROC-AUC en
       validation croisée à 5 plis. Une variable seule proche de 1 trahit
       presque toujours une information postérieure à la date de prédiction
       (ex. un « coût de réparation » qui n'existe que si la panne a eu lieu).
       L'arbre, contrairement à une corrélation, capte aussi les relations
       non linéaires (effet de seuil, en U).
    """
    # One-hot : l'arbre a besoin de colonnes numériques ; chaque modalité
    # (ex. type_pompe_manuelle) est testée séparément.
    X = pd.get_dummies(hist[FEATURES], columns=CAT_FEATURES)
    y = hist[TARGET]
    auc_univ = {
        c: cross_val_score(
            DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
            X[[c]], y, cv=5, scoring="roc_auc",
        ).mean()
        for c in X.columns
    }
    return {
        "doublons_hist": int(hist.duplicated().sum()),
        # Jointure interne sur toutes les colonnes : lignes strictement identiques.
        "recouvrement_hist_new": int(len(hist.merge(new, how="inner"))),
        "auc_univariee": pd.Series(auc_univ).sort_values(ascending=False).round(3),
    }


def main():
    hist, new = load()
    REPORTS.mkdir(exist_ok=True)
    fuite = audit_fuite(hist, new)

    # Le rapport est assemblé en Markdown pour être lisible directement sur
    # GitHub et versionné avec le code qui le produit.
    parts = [
        "# Exploration des données\n",
        f"Historique : {hist.shape[0]} lignes ; nouveau mois : {new.shape[0]} lignes.\n",
        "## Qualité (historique)\n", qualite(hist).to_markdown(), "\n",
        # describe() : effectif, moyenne, écart-type, min, quartiles, max.
        # Repère les valeurs aberrantes (ex. débit négatif, âge de 200 ans).
        "## Statistiques descriptives (historique)\n",
        hist[NUM_FEATURES].describe().T.round(2).to_markdown(), "\n",
        # Taux de positifs : mesure le déséquilibre des classes et fixe la
        # valeur de la PR-AUC d'un modèle au hasard.
        f"## Cible\nTaux de panne historique : {hist[TARGET].mean():.3f} ; "
        f"nouveau mois : {new[TARGET].mean():.3f}\n",
        "## Taux de panne par modalité\n",
        taux_par_modalite(hist, "type_pompe").to_markdown(), "\n",
        taux_par_modalite(hist, "region").to_markdown(), "\n",
        # pannes_12m est numérique mais ne prend que quelques valeurs :
        # le taux par valeur montre la forme de la relation (croissante ?).
        taux_par_modalite(hist, "pannes_12m").to_markdown(), "\n",
        # Corrélation de Pearson avec la cible 0/1 : force et sens d'une
        # relation LINÉAIRE uniquement. Proche de 0 ne veut pas dire « inutile »
        # (relation possible non linéaire) ; l'AUC univariée complète ce test.
        "## Corrélation avec la cible\n",
        hist[NUM_FEATURES + [TARGET]].corr()[TARGET].drop(TARGET)
        .round(3).to_frame("corr").to_markdown(), "\n",
        "## Historique vs nouveau mois\n", comparaison(hist, new).to_markdown(), "\n",
        "## Audit de fuite\n",
        f"- Doublons dans l'historique : {fuite['doublons_hist']}\n",
        f"- Lignes communes historique / nouveau mois : {fuite['recouvrement_hist_new']}\n",
        "\nAUC de chaque variable seule (arbre de profondeur 3, CV 5 plis) :\n\n",
        fuite["auc_univariee"].to_frame("auc").to_markdown(), "\n",
    ]
    out = REPORTS / "eda.md"
    out.write_text("\n".join(parts), encoding="utf-8")

    # Résumé console : les deux constats clés (dérive, absence de fuite).
    print(f"Rapport écrit : {out}")
    print(comparaison(hist, new).to_string(index=False))
    print("\nAUC univariée max :", fuite["auc_univariee"].iloc[0],
          f"({fuite['auc_univariee'].index[0]})")


if __name__ == "__main__":
    main()