# Prédiction des pannes de forages à 90 jours

Service estimant le risque de panne de chaque forage dans les 90 jours,
pour planifier les visites préventives.

## Données
- `data/raw/forages_historique.csv` : 3 000 forages, cible `panne_90j`
- `data/raw/forages_nouveau_mois.csv` : 600 forages, mois récent simulé

## Installation
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements-lock.txt

## Étapes
- [x] Initialisation du dépôt
- [ ] Exploration des données
- [ ] Entraînement tracé (MLflow)
- [ ] API (FastAPI)
- [ ] Tests (pytest)
- [ ] Conteneurisation (Docker)
- [ ] CI (GitHub Actions)
- [ ] Monitoring et ré-entraînement