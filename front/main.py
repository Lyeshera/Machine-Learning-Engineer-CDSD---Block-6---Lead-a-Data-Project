import requests
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Tuple, List, Dict, Any
import plotly.express as px
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import plotly
import json

# URL de l'API pour les prédictions
API_URL = "https://forex-api-jedha-back-14619c7a835e.herokuapp.com/predict"
# Chemin du fichier CSV contenant les données
CSV_FILE_PATH = "EURLBPX2.csv"
# Nombre de pas de temps pour les séquences
TIME_STEP = 60

# Initialisation de l'application FastAPI
app = FastAPI()

# Fonction pour charger et prétraiter les données
def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)
    data["Date"] = pd.to_datetime(data["Date"])
    data.set_index("Date", inplace=True)
    return data

# Fonction pour ajouter des caractéristiques au dataframe
def add_features(data: pd.DataFrame) -> pd.DataFrame:
    # Calcul des retours logarithmiques
    data["Log Returns"] = np.log(data["Close"] / data["Close"].shift(1))
    # Moyennes mobiles sur différentes périodes
    data["3D MA"] = data["Close"].rolling(window=3).mean()
    data["7D MA"] = data["Close"].rolling(window=7).mean()
    data["15D MA"] = data["Close"].rolling(window=15).mean()
    data["30D MA"] = data["Close"].rolling(window=30).mean()
    # Bandes de Bollinger sur différentes périodes
    data["Bollinger High 7D"] = data["7D MA"] + (data["Close"].rolling(window=7).std() * 2)
    data["Bollinger Low 7D"] = data["7D MA"] - (data["Close"].rolling(window=7).std() * 2)
    data["Bollinger High 30D"] = data["30D MA"] + (data["Close"].rolling(window=30).std() * 2)
    data["Bollinger Low 30D"] = data["30D MA"] - (data["Close"].rolling(window=30).std() * 2)
    # Indicateurs techniques RSI, MACD, ATR
    data["RSI"] = ta.momentum.rsi(data["Close"], window=14)
    data["MACD"] = ta.trend.macd(data["Close"])
    data["MACD Signal"] = ta.trend.macd_signal(data["Close"])
    data["MACD Hist"] = ta.trend.macd_diff(data["Close"])
    data["ATR"] = ta.volatility.average_true_range(data["High"], data["Low"], data["Close"], window=14)
    # Décomposition saisonnière de la série temporelle
    decomposition = seasonal_decompose(data["Close"], model="multiplicative", period=30)
    data["Trend"] = decomposition.trend
    data["Seasonal"] = decomposition.seasonal
    data["Resid"] = decomposition.resid
    # Ajout des composants jour, mois, année, et des décalages
    data["Day"] = data.index.day
    data["Month"] = data.index.month
    data["Year"] = data.index.year
    data["Lag1"] = data["Close"].shift(1)
    data["Lag2"] = data["Close"].shift(2)
    data.dropna(inplace=True)
    return data

# Fonction pour normaliser les données
def normalize_data(data: pd.DataFrame, features: List[str]) -> np.ndarray:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[features])
    return scaled_data

# Fonction pour diviser les données en ensembles d'entraînement et de test
def split_data(data: np.ndarray, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

# Fonction pour créer un ensemble de données avec des séquences de temps
def create_dataset(dataset: np.ndarray, time_step: int) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i : (i + time_step)])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

# Fonction pour ajouter une caractéristique supplémentaire à l'ensemble de données
def add_additional_feature(X: np.ndarray) -> np.ndarray:
    additional_feature = np.zeros((X.shape[0], X.shape[1], 1))
    X = np.concatenate((X, additional_feature), axis=2)
    return X

# Fonction pour obtenir des prédictions à partir de l'API
def get_predictions(api_url: str, payload: Dict[str, Any]) -> List[float]:
    response = requests.post(api_url, json=payload)
    predictions = response.json()['prediction']
    return [item[0] for item in predictions]

# Fonction pour tracer les prédictions et les données réelles
def plot_predictions(train_actual: np.ndarray, train_predict: List[float], test_actual: np.ndarray, test_predict: List[float], time_step: int) -> Tuple[str, str]:
    train_actual = train_actual[time_step:time_step + len(train_predict)]
    test_actual = test_actual[:len(test_predict)]
    
    train_df = pd.DataFrame({
        'Time': range(time_step, time_step + len(train_predict)),
        'Actual': train_actual,
        'Predicted': train_predict
    })

    test_df = pd.DataFrame({
        'Time': range(len(train_actual) + time_step, len(train_actual) + time_step + len(test_predict)),
        'Actual': test_actual,
        'Predicted': test_predict
    })

    fig_train = px.line(train_df, x='Time', y=['Actual', 'Predicted'], title='Actual vs. Predicted Prices - Training Data')
    fig_train.update_layout(yaxis_title='Normalized Price', xaxis_title='Time')

    fig_test = px.line(test_df, x='Time', y=['Actual', 'Predicted'], title='Actual vs. Predicted Prices - Testing Data')
    fig_test.update_layout(yaxis_title='Normalized Price', xaxis_title='Time')

    train_html = fig_train.to_html(full_html=False)
    test_html = fig_test.to_html(full_html=False)

    return train_html, test_html

# Endpoint FastAPI pour afficher le tableau de bord avec les prédictions
@app.get("/dashboard", response_class=HTMLResponse)
def predict():
    # Charge et prétraite les données
    data = load_and_preprocess_data(CSV_FILE_PATH)
    data = add_features(data)

    # Caractéristiques utilisées pour la normalisation
    features = ["Close", "RSI", "MACD", "MACD Signal", "MACD Hist", "Lag1", "Lag2", "7D MA", "30D MA", "Bollinger High 7D", "Bollinger Low 7D", "ATR", "Log Returns"]
    scaled_data = normalize_data(data, features)
    
    # Divise les données en ensembles d'entraînement et de test
    train_data, test_data = split_data(scaled_data)
    train_size = int(len(scaled_data) * 0.8)
    
    # Crée des ensembles de données avec des séquences de temps
    X_train, y_train = create_dataset(train_data, TIME_STEP)
    X_test, y_test = create_dataset(test_data, TIME_STEP)

    # Ajoute des caractéristiques supplémentaires
    X_train = add_additional_feature(X_train)
    X_test = add_additional_feature(X_test)
    
    # Prépare les charges utiles pour les prédictions
    train_payload = {"input": X_train.tolist()}
    test_payload = {"input": X_test.tolist()}
    
    # Obtient les prédictions de l'API
    train_predict = get_predictions(API_URL, train_payload)
    test_predict = get_predictions(API_URL, test_payload)

    # Données réelles pour l'entraînement et le test
    train_actual = scaled_data[:train_size, 0]
    test_actual = scaled_data[train_size:, 0]

    # Trace les prédictions et les données réelles
    train_html, test_html = plot_predictions(train_actual, train_predict, test_actual, test_predict, TIME_STEP)

    # Crée le contenu HTML pour le tableau de bord
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Predictions</title>
    </head>
    <body>
        <h1>Training Data</h1>
        {train_html}
        <h1>Testing Data</h1>
        {test_html}
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Point d'entrée pour exécuter l'application FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
