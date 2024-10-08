from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay,RocCurveDisplay
from datetime import datetime


class ModelEvaluator:
    def evaluate_model(self, model, x_val, y_val):
        # Fazer as previsões
        y_pred = model.predict(x_val)

        # Calcular a matriz de confusão
        cm = confusion_matrix(y_val, y_pred)
        print("Matriz de Confusão:\n", cm)

        # Gerar as métricas diretamente da matriz de confusão
        metrics_report = classification_report(y_val, y_pred, output_dict=True)

        # Exibir o relatório das métricas
        print("Relatório de Métricas:\n", classification_report(y_val, y_pred))

        # Exibir e retornar as métricas mais detalhadas
        return {
            "accuracy": metrics_report["accuracy"],
            "precision": metrics_report["weighted avg"]["precision"],
            "recall": metrics_report["weighted avg"]["recall"],
            "f1_score": metrics_report["weighted avg"]["f1-score"],
            "roc_auc": metrics_report.get(
                "roc_auc", None
            ),  # ROC AUC só está disponível para classificadores binários
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
