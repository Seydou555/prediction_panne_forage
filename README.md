# Prédiction des pannes de forages à 90 jours

Service estimant le risque de panne de chaque forage dans les 90 jours,
pour planifier les visites préventives.

## Données
- `data/forages_historique.csv` : 3 000 forages, cible `panne_90j`
- `data/forages_nouveau_mois.csv` : 600 forages, mois récent simulé

## Installation
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements-lock.txt

## Étapes
- [x] Initialisation du dépôt
- [X] Exploration des données
- [X] Entraînement tracé (MLflow)
- [ ] API (FastAPI)
- [ ] Tests (pytest)
- [ ] Conteneurisation (Docker)
- [ ] CI (GitHub Actions)
- [ ] Monitoring et ré-entraînement

## Lancez le script d'entraînement pour générer les artefacts
-python -m src.entrainement

## Exécutez le serveur Uvicorn en local
uvicorn src.api.main:app --reload

## Conteneurisation(Docker)
Construction de l'image Docker
docker build -t panne-forage-api .

## Lancement du conteneur
Construction de l'image Docker
docker build -t panne-forage-api .

## Lancement du conteneur
-Construction de l'image Docker: docker build -t panne-forage-api
-Lancement du conteneur: docker run -p 8000:8000 panne-forage-api