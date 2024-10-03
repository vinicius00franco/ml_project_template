from sklearn.preprocessing import label_binarize

class Trainer:
    def __init__(self, data_loader, model_builder, evaluator):
        self.data_loader = data_loader
        self.model_builder = model_builder
        self.evaluator = evaluator

    def train_and_evaluate(self):
        data = self.data_loader.load_data()
        x = data.drop('inadimplente', axis=1)
        y = data['inadimplente']
        x_treino, x_val, x_teste, y_treino, y_val, y_teste = self.data_loader.split_data(x, y)

        model = self.model_builder.train_model(x_treino, y_treino)

        acc_train = model.score(x_treino, y_treino)
        acc_val, f1_val,accuracy_score_val,precision_val,recall_val,roc_auc  = self.evaluator.evaluate_model(model, x_val, y_val)

        print(f'Acurácia de treino: {acc_train}')
        print(f'Acurácia de validação: {acc_val}')
        print(f'F1 Score: {f1_val}')
        print(f'Acurácia Score: {accuracy_score_val}')
        
        print(f'Precisão: {precision_val}')
        print(f'Recall: {recall_val}')
        
        print(f'AUC: {roc_auc}')
        
        y_pred = model.predict(x_val)
        
        # Binarizar y_val e y_pred
        y_val_bin = label_binarize(y_val, classes=[0, 1])
        y_pred_bin = label_binarize(y_pred, classes=[0, 1])

        

        
        self.evaluator.plot_confusion_matrix(y_val, y_pred)
        # Corrigir chamada do método plot_roc
        self.evaluator.plot_roc(y_val_bin, y_pred_bin)
