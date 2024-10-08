from fastapi import FastAPI
from schema import InputData, Prediction
from model_evaluator import ModelEvaluator
import numpy as np

app = FastAPI()

# Carregar o modelo ao iniciar a API
evaluator = ModelEvaluator(model_path="models/modelo_treinado.pkl")

@app.post("/predict", response_model=Prediction)
async def predict(data: InputData):
    prediction = evaluator.predict([data.features])
    return {"label": int(prediction[0]), "confidence": 0.9}  # Coloque a confiança aqui
