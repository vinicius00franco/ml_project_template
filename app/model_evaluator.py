from sklearn.metrics import classification_report
from model_loader import ModelLoader

class ModelEvaluator:
    def __init__(self, model_path):
        self.model = ModelLoader.load_model(model_path)

    def predict(self, data):
        return self.model.predict(data)

    def evaluate(self, x_val, y_val):
        y_pred = self.model.predict(x_val)
        return classification_report(y_val, y_pred, output_dict=True)
