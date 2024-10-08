import joblib
import os
from datetime import datetime

class ModelSaver:
    @staticmethod
    def save_model(model, model_type, balance_strategy, filename=None):
        # Criar um nome de arquivo com base no modelo e estratégia
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        
        # Criar um nome de arquivo com base no modelo, estratégia e timestamp
        if filename is None:
            filename = f"models/{model_type}_{balance_strategy}_{timestamp}_trained_model.pkl"
        
        
        # Criar o diretório, caso não exista
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Salvar o modelo no arquivo
        joblib.dump(model, filename)
        print(f"Modelo salvo em: {filename}")

    @staticmethod
    def load_model(filename="models/trained_model.pkl"):
        return joblib.load(filename)
