from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    RocCurveDisplay,
    precision_recall_curve,
    roc_auc_score,
    classification_report
)
import matplotlib.pyplot as plt


class ModelEvaluator:
    @staticmethod
    def evaluate_model(model, x_val, y_val):
        y_pred = model.predict(x_val)
        acc = model.score(x_val, y_val)
        f1 = f1_score(y_val, y_pred)
        acc_score = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        
        roc_auc = roc_auc_score(y_val, y_pred)
        
        

        return acc, f1, acc_score, precision, recall, roc_auc

    @staticmethod
    def plot_confusion_matrix(y_val, y_pred, filename="graficos/matriz_confusao.png"):
        matriz_confusao = confusion_matrix(y_val, y_pred)
        fig, ax = plt.subplots()
        ax.set_title("Matriz de Confusão")

        im = ax.imshow(matriz_confusao, cmap="Blues")
        for i in range(matriz_confusao.shape[0]):
            for j in range(matriz_confusao.shape[1]):
                ax.text(
                    j,
                    i,
                    matriz_confusao[i, j],
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=16,
                )

        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.xticks([0, 1], ["Adimplente", "Inadimplente"])
        plt.yticks([0, 1], ["Adimplente", "Inadimplente"])
        plt.colorbar(im)
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def plot_roc(y_val, y_pred, filename="graficos/curva_roc.png"):
        
        RocCurveDisplay.from_predictions(y_val, y_pred, name = 'Árvore de Decisão');
        plt.title("Curva ROC")
        plt.savefig(filename)
        plt.close()  

    @staticmethod
    def print_classification_report(y_val, y_pred):
        report = classification_report(y_val, y_pred)
        print(report)
        

        
