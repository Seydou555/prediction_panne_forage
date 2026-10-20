"""Exploration des données et audit de fuite.

Usage : python -m src.eda
Produit : reports/eda.md
"""
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    CAT_FEATURES, FEATURES, HIST_PATH, NEW_PATH, NUM_FEATURES,
    RANDOM_STATE, REPORTS, TARGET,
)


def load():
    return pd.read_csv(HIST_PATH), pd.read_csv(NEW_PATH)


def qualite(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "type": df.dtypes.astype(str),
        "manquants": df.isna().sum(),
        "uniques": df.nunique(),
    })


def comparaison(hist: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Statistiques historiques vs nouveau mois : premier signal de dérive."""
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
    return df.groupby(col)[TARGET].agg(taux_panne="mean", effectif="size").round(3)


def audit_fuite(hist: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Recherche de signaux de fuite de données.

    1. Doublons internes et recouvrement historique / nouveau mois.
    2. Pouvoir prédictif de chaque variable seule (AUC en validation croisée).
       Une variable seule avec une AUC proche de 1 trahit presque toujours
       une information postérieure à la date de prédiction.
    """
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
        "recouvrement_hist_new": int(len(hist.merge(new, how="inner"))),
        "auc_univariee": pd.Series(auc_univ).sort_values(ascending=False).round(3),
    }


def main():
    hist, new = load()
    REPORTS.mkdir(exist_ok=True)
    fuite = audit_fuite(hist, new)

    parts = [
        "# Exploration des données\n",
        f"Historique : {hist.shape[0]} lignes ; nouveau mois : {new.shape[0]} lignes.\n",
        "## Qualité (historique)\n", qualite(hist).to_markdown(), "\n",
        "## Statistiques descriptives (historique)\n",
        hist[NUM_FEATURES].describe().T.round(2).to_markdown(), "\n",
        f"## Cible\nTaux de panne historique : {hist[TARGET].mean():.3f} ; "
        f"nouveau mois : {new[TARGET].mean():.3f}\n",
        "## Taux de panne par modalité\n",
        taux_par_modalite(hist, "type_pompe").to_markdown(), "\n",
        taux_par_modalite(hist, "region").to_markdown(), "\n",
        taux_par_modalite(hist, "pannes_12m").to_markdown(), "\n",
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
    print(f"Rapport écrit : {out}")
    print(comparaison(hist, new).to_string(index=False))
    print("\nAUC univariée max :", fuite["auc_univariee"].iloc[0],
          f"({fuite['auc_univariee'].index[0]})")


if __name__ == "__main__":
    main()