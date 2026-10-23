# Utiliser une image Python officielle légère
FROM python:3.13-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier des dépendances en premier pour optimiser le cache Docker
COPY requirements.txt .

# Installer les dépendances Python nécessaires
RUN pip install --no-cache-dir -r requirements.txt

# Copier l'ensemble du code source et des artefacts (dossiers src/, models/, etc.)
COPY . .

# Exposer le port sur lequel l'application s'exécute
EXPOSE 8000

# Commande pour démarrer l'API FastAPI avec Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]