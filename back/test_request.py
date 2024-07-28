import requests

# URL de l'API pour les prédictions
url = "https://forex-api-jedha-back-14619c7a835e.herokuapp.com/predict"

# Exemple d'entrée pour l'API
sample_input = {
    "input": [ # "input" est une liste contenant une séquence de 60 pas de temps,
        [[0.1]*14]*60 # chaque pas de temps ayant 14 caractéristiques (ici, chaque caractéristique est 0.1)
    ]
}

# Envoi d'une requête POST à l'API avec les données d'entrée sous forme de JSON
response = requests.post(url, json=sample_input)

# Affichage de la réponse JSON renvoyée par l'API
print(response.json())
