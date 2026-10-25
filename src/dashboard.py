"""Tableau de bord Streamlit : forages à risque, registre MLflow, surveillance.

Usage (depuis la racine du dépôt) : streamlit run src/dashboard.py

Prérequis : python -m src.entrainement (modèle + registre MLflow).
Optionnel  : python -m src.monitoring.drift (onglet Surveillance).
"""
import sys
from pathlib import Path

# Streamlit ajoute src/ au chemin d'import, pas la racine : on l'ajoute
# pour que les imports "from src..." fonctionnent.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json  # noqa: E402

import joblib  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import FEATURES, NEW_PATH, REPORTS, ROOT  # noqa: E402
from src.entrainement import EXPERIMENT, REGISTERED_NAME, TRACKING_URI  # noqa: E402

MODELS_DIR = ROOT / "models"

st.set_page_config(page_title="Pannes de forages", page_icon="💧", layout="wide")


# --- Chargements (mis en cache : relus seulement si le fichier change) -------------

@st.cache_resource
def charger_modele(mtime: float):
    """mtime en argument : un nouveau model.joblib invalide le cache."""
    modele = joblib.load(MODELS_DIR / "model.joblib")
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
    return modele, metadata


@st.cache_data
def scorer(df: pd.DataFrame, _modele, cle_modele: str) -> pd.DataFrame:
    """Ajoute la probabilité de panne et un identifiant à chaque forage."""
    df = df.copy()
    if "forage_id" not in df.columns:  # les données fournies n'ont pas d'identifiant
        df.insert(0, "forage_id", [f"F-{i + 1:04d}" for i in range(len(df))])
    df["probabilite_panne"] = _modele.predict_proba(df[FEATURES])[:, 1]
    return df


def client_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.MlflowClient()


# --- Onglet 1 : forages à risque -------------------------------------------------------

def onglet_risque():
    chemin_modele = MODELS_DIR / "model.joblib"
    if not chemin_modele.exists():
        st.error("Aucun modèle trouvé. Lancer d'abord : python -m src.entrainement")
        return
    modele, metadata = charger_modele(chemin_modele.stat().st_mtime)

    st.sidebar.header("Données")
    fichier = st.sidebar.file_uploader("Fichier CSV de forages (optionnel)", type="csv")
    source = pd.read_csv(fichier) if fichier else pd.read_csv(NEW_PATH)
    manquantes = set(FEATURES) - set(source.columns)
    if manquantes:
        st.error(f"Colonnes manquantes dans le fichier : {sorted(manquantes)}")
        return
    st.sidebar.caption("Source : " + (fichier.name if fichier else NEW_PATH.name))

    st.sidebar.header("Filtres")
    seuil = st.sidebar.slider("Seuil d'alerte", 0.05, 0.95, float(metadata["seuil"]), 0.01,
                              help=f"Seuil du modèle en production : {metadata['seuil']}")
    regions = st.sidebar.multiselect("Région", sorted(source.region.unique()),
                                     default=sorted(source.region.unique()))
    pompes = st.sidebar.multiselect("Type de pompe", sorted(source.type_pompe.unique()),
                                    default=sorted(source.type_pompe.unique()))
    capacite = st.sidebar.number_input("Visites possibles sur la période", 1, 5000, 50,
                                       help="Nombre de visites préventives que les équipes "
                                            "peuvent réaliser : les plus risqués d'abord.")

    df = scorer(source, modele, str(metadata.get("mlflow_run_id")))
    df = df[df.region.isin(regions) & df.type_pompe.isin(pompes)]
    df["a_risque"] = df.probabilite_panne >= seuil
    risque = df[df.a_risque].sort_values("probabilite_panne", ascending=False)

    st.subheader("Forages à risque de panne dans les 90 jours")
    st.caption(f"Modèle {metadata['modele']} - version {metadata.get('registre_version')} "
               f"du registre - entraîné le {metadata.get('entraine_le', '?')[:10]}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forages suivis", len(df))
    c2.metric("Forages à risque", len(risque),
              f"{len(risque) / max(len(df), 1):.0%} du parc", delta_color="off")
    c3.metric("Probabilité moyenne", f"{df.probabilite_panne.mean():.2f}")
    c4.metric("Visites planifiables", min(capacite, len(risque)))
    if len(risque) > capacite:
        st.warning(f"{len(risque)} forages dépassent le seuil pour {capacite} visites possibles : "
                   f"la liste ci-dessous est triée par risque décroissant, les "
                   f"{capacite} premiers sont prioritaires.")

    gauche, droite = st.columns(2)
    with gauche:
        st.markdown("**Forages à risque par région**")
        st.bar_chart(risque.groupby("region").size().rename("forages"))
    with droite:
        st.markdown("**Forages à risque par type de pompe**")
        st.bar_chart(risque.groupby("type_pompe").size().rename("forages"))

    st.markdown("**Liste de priorité des visites**")
    liste = risque.head(capacite).copy()
    liste.insert(1, "priorite", range(1, len(liste) + 1))
    colonnes = ["forage_id", "priorite", "probabilite_panne", "region", "type_pompe",
                "age_ans", "pannes_12m", "jours_depuis_maintenance", "usagers_jour"]
    st.dataframe(
        liste[colonnes], hide_index=True, width="stretch",
        column_config={"probabilite_panne": st.column_config.ProgressColumn(
            "Probabilité de panne", min_value=0.0, max_value=1.0, format="%.2f")},
    )
    st.download_button("Télécharger la liste (CSV)", liste[colonnes].to_csv(index=False),
                       "visites_prioritaires.csv", "text/csv")


# --- Onglet 2 : registre MLflow -----------------------------------------------------------

def onglet_registre():
    if not (ROOT / "mlflow.db").exists():
        st.error("Base MLflow introuvable. Lancer d'abord : python -m src.entrainement")
        return
    client = client_mlflow()
    try:
        versions = client.search_model_versions(f"name='{REGISTERED_NAME}'")
    except mlflow.exceptions.MlflowException:
        versions = []
    if not versions:
        st.warning(f"Aucune version du modèle « {REGISTERED_NAME} » dans le registre.")
        return

    # Les alias ne sont pas toujours renvoyés par search_model_versions :
    # on interroge directement l'alias "production".
    try:
        v_prod = client.get_model_version_by_alias(REGISTERED_NAME, "production").version
    except mlflow.exceptions.MlflowException:
        v_prod = None

    lignes = []
    for v in sorted(versions, key=lambda v: int(v.version), reverse=True):
        m = client.get_run(v.run_id).data.metrics
        lignes.append({
            "version": int(v.version),
            "alias": "production" if v.version == v_prod else "-",
            "seuil": v.tags.get("seuil", "-"),
            "cree_le": pd.to_datetime(v.creation_timestamp, unit="ms").strftime("%Y-%m-%d %H:%M"),
            "cv_pr_auc": m.get("cv_pr_auc"),
            "test_pr_auc": m.get("test_pr_auc", m.get("test_historique_pr_auc")),
            "test_rappel": m.get("test_rappel", m.get("test_historique_rappel")),
            "test_precision": m.get("test_precision", m.get("test_historique_precision")),
            "run_id": v.run_id,
        })
    registre = pd.DataFrame(lignes)

    prod = registre[registre.alias.str.contains("production")]
    st.subheader(f"Registre MLflow : {REGISTERED_NAME}")
    if not prod.empty:
        p = prod.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Version en production", p.version)
        c2.metric("Seuil", p.seuil)
        c3.metric("PR-AUC (test)", f"{p.test_pr_auc:.3f}" if pd.notna(p.test_pr_auc) else "-")
        c4.metric("Rappel (test)", f"{p.test_rappel:.3f}" if pd.notna(p.test_rappel) else "-")
    st.markdown("**Versions enregistrées**")
    st.dataframe(registre, hide_index=True, width="stretch")

    st.markdown(f"**Tous les runs de l'expérience « {EXPERIMENT} »**")
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT])
    if not runs.empty:
        cols = ["tags.mlflow.runName", "start_time", "params.seuil"] + sorted(
            c for c in runs.columns if c.startswith("metrics.")
            and any(k in c for k in ("pr_auc", "rappel", "precision", "gain")))
        runs = runs[[c for c in cols if c in runs.columns]].rename(
            columns=lambda c: c.split(".")[-1]).rename(columns={"runName": "run"})
        st.dataframe(runs.round(3), hide_index=True, width="stretch")
    st.caption("Détail complet : mlflow ui --backend-store-uri sqlite:///mlflow.db")


# --- Onglet 3 : surveillance --------------------------------------------------------------

def onglet_surveillance():
    chemin = REPORTS / "drift.json"
    if not chemin.exists():
        st.info("Pas encore de rapport. Lancer : python -m src.monitoring.drift")
        return
    rapport = json.loads(chemin.read_text(encoding="utf-8"))
    dec, perf = rapport["decision"], rapport["production"]

    st.subheader("Surveillance du modèle en production")
    st.caption(f"Rapport du {rapport['date'][:16].replace('T', ' ')} UTC")
    if dec["reentrainer"]:
        st.error("Ré-entraînement recommandé : " + " ; ".join(dec["raisons"]))
    else:
        st.success("Aucune dérive ni baisse de performance au-delà des seuils.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Taux d'alerte", f"{perf['taux_alerte_nouveau_mois']:.0%}",
              f"{perf['taux_alerte_nouveau_mois'] - perf['taux_alerte_reference']:+.0%}",
              delta_color="inverse")
    c2.metric("Rappel (nouveau mois)", f"{perf['nouveau_mois']['rappel']:.2f}",
              f"{perf['nouveau_mois']['rappel'] - perf['reference']['rappel']:+.2f}")
    c3.metric("ROC-AUC (nouveau mois)", f"{perf['nouveau_mois']['roc_auc']:.2f}",
              f"{perf['nouveau_mois']['roc_auc'] - perf['reference']['roc_auc']:+.2f}")

    st.markdown("**Dérive des variables (PSI : < 0,10 stable · 0,10–0,20 modérée · > 0,20 forte)**")
    derive = pd.DataFrame(rapport["derive_donnees"])
    st.bar_chart(derive.set_index("variable")["psi"])
    st.dataframe(derive, hide_index=True, width="stretch")

    if "reentrainement" in rapport:
        r = rapport["reentrainement"]
        st.markdown("**Dernier ré-entraînement**")
        st.write(f"Gain de PR-AUC : {r['gain_pr_auc']:+.3f} · "
                 + ("promu en production" if r["promu"] else "non promu, ancien modèle conservé"))


onglet1, onglet2, onglet3 = st.tabs(["Forages à risque", "Registre MLflow", "Surveillance"])
with onglet1:
    onglet_risque()
with onglet2:
    onglet_registre()
with onglet3:
    onglet_surveillance()
