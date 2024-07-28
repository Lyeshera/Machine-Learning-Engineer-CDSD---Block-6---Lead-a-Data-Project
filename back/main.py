from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Initialisation de l'application FastAPI
app = FastAPI()

# Chemin vers le modèle pré-entraîné
model_path = 'eurlbp_lstm_model.h5'
# Chargement du modèle Keras pré-entraîné
model = load_model(model_path)

# Définition de la structure de la requête de prédiction avec Pydantic
class PredictRequest(BaseModel):
    input: list

# Fonction de prétraitement pour valider et transformer les données d'entrée
def preprocess_input(input_data, time_steps=60):
    input_array = np.array(input_data)
    if input_array.shape[1:] != (time_steps, 14):
        # Validation de la forme des données d'entrée
        raise ValueError("Input array must have shape (samples, 60, 14)")
    return input_array

# Définition de l'endpoint POST pour les prédictions
@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        # Prétraitement des données d'entrée
        input_data = preprocess_input(request.input)
    except ValueError as e:
        # Retourne une erreur si les données d'entrée ne sont pas valides
        return {"error": str(e)}
    
    # Réalisation de la prédiction à l'aide du modèle chargé
    prediction = model.predict(input_data)
    # Retourne les prédictions sous forme de liste
    return {"prediction": prediction.tolist()}

# Point d'entrée pour exécuter l'application avec Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
