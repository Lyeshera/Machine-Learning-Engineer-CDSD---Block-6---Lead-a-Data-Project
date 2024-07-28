# Utilise une image de base Miniconda3 de ContinuumIO
FROM continuumio/miniconda3

# Définit le répertoire de travail dans le conteneur
WORKDIR /home/app

# Met à jour les paquets et installe des outils nécessaires
RUN apt-get update -y \
    && apt-get install -y \
    nano \
    unzip \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Crée un nouvel environnement Conda nommé fastapi-env avec Python 3.8
RUN conda create -n fastapi-env python=3.8 -y

# Définit l'environnement par défaut du shell comme étant celui de fastapi-env
SHELL ["conda", "run", "-n", "fastapi-env", "/bin/bash", "-c"]

# Installe pip dans l'environnement Conda fastapi-env
RUN conda install pip -y

# Installe gunicorn via pip dans l'environnement Conda fastapi-env
RUN conda run -n fastapi-env pip install gunicorn

# Copie le fichier requirements.txt dans le répertoire de travail
COPY requirements.txt .

# Installe les dépendances listées dans requirements.txt via pip dans l'environnement Conda fastapi-env
RUN conda run -n fastapi-env pip install --no-cache-dir -r requirements.txt

# Copie tous les fichiers du répertoire local actuel dans le répertoire de travail du conteneur
COPY . .

# Expose le port 8000 pour le conteneur
EXPOSE 8000

# Démarre l'application FastAPI avec gunicorn, en utilisant l'option --worker-class pour spécifier le worker uvicorn
CMD gunicorn main:app --bind 0.0.0.0:$PORT --timeout 100 --worker-class uvicorn.workers.UvicornWorker -n fastapi-env
