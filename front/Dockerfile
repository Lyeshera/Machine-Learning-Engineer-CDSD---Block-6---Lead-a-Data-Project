# Utilise l'image Miniconda
FROM continuumio/miniconda3

# Définit le répertoire de travail dans le conteneur
WORKDIR /home/app

# Met à jour les paquets et installe des outils nécessaires, puis nettoie le cache des paquets pour réduire la taille de l'image
RUN apt-get update -y \
    && apt-get install -y \
    nano \
    unzip \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Crée un nouvel environnement Conda et définit cet environnement comme le shell par défaut
RUN conda create -n fastapi-env python=3.8 -y
SHELL ["conda", "run", "-n", "fastapi-env", "/bin/bash", "-c"]

# Installe pip et gunicorn dans l'environnement Conda fastapi-env
RUN conda install pip -y \
    && conda run -n fastapi-env pip install gunicorn

# Copie le fichier requirements.txt dans le répertoire de travail et installe les dépendances listées
COPY requirements.txt .
RUN conda run -n fastapi-env pip install --no-cache-dir -r requirements.txt

# Copie tous les fichiers du répertoire local actuel dans le répertoire de travail du conteneur
COPY . .

# Expose le port 8000 pour permettre l'accès à l'application
EXPOSE 8000

# Démarre l'application FastAPI avec gunicorn, liant l'application à toutes les interfaces réseau sur le port spécifié par $PORT
# et Utilise uvicorn comme worker et définit un timeout de 100 secondes
CMD gunicorn main:app --bind 0.0.0.0:$PORT --timeout 100 --worker-class uvicorn.workers.UvicornWorker -n fastapi-env
