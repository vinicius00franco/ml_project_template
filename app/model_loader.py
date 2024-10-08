import joblib

class ModelLoader:
    @staticmethod
    def load_model(model_path):
        return joblib.load(model_path)
