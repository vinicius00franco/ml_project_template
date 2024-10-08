from sklearn.metrics import (
    confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from datetime import datetime


class ModelEvaluator:
    def evaluate_model(self, model, x_val, y_val):
        y_pred = model.predict(x_val)
        return {
            "accuracy": accuracy_score(y_val, y_pred),
            "f1_score": f1_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred),
            "roc_auc": roc_auc_score(y_val, y_pred)
        }
    
    
    

    def plot_confusion_matrix(self, y_val, y_pred, filename=None):
        # Obter data e hora atual
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        
        # Gerar nome do arquivo com timestamp se não fornecido
        if filename is None:
            filename = f"graficos/matriz_confusao_{timestamp}.png"
        
        # Plotar a matriz de confusão
        ConfusionMatrixDisplay.from_predictions(y_val, y_pred)
        
        # Salvar o gráfico no arquivo
        plt.savefig(filename)
        plt.close()
        print(f"Matriz de confusão salva em: {filename}")

    def plot_roc_curve(self, y_val, y_pred, filename=None):
        # Obter data e hora atual
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        
        # Gerar nome do arquivo com timestamp se não fornecido
        if filename is None:
            filename = f"graficos/curva_roc_{timestamp}.png"
        
        # Plotar a curva ROC
        RocCurveDisplay.from_predictions(y_val, y_pred)
        
        # Salvar o gráfico no arquivo
        plt.savefig(filename)
        plt.close()
        print(f"Curva ROC salva em: {filename}")
        
        

        
