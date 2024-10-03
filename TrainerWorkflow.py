from sklearn.preprocessing import label_binarize

class TrainerWorkflow:
    def __init__(self, data_loader, model_builder, evaluator):
        self.data_loader = data_loader
        self.model_builder = model_builder
        self.evaluator = evaluator

    def train_and_evaluate(self):
        # Carregar e preparar os dados
        data = self.data_loader.load_data()
        x = data.drop('inadimplente', axis=1)
        y = data['inadimplente']
        
        # Dividir os dados em treino, validação e teste
        x_treino, x_val, x_teste, y_treino, y_val, y_teste = self.data_loader.split_data(x, y)

        # Treinar o modelo
        model = self.model_builder.train_model(x_treino, y_treino)
        
        # Cross-validation para avaliar o modelo
        cv_results = self.model_builder.cross_validate_model(x, y)
        media = cv_results['test_score'].mean()
        desvio_padrao = cv_results['test_score'].std()
        print(f'Intervalo de confiança: [{media - 2*desvio_padrao}, {min(media + 2*desvio_padrao, 1)}]')

        # Avaliar o modelo nos dados de treino e validação
        acc_train = model.score(x_treino, y_treino)
        acc_val, f1_val, accuracy_score_val, precision_val, recall_val, roc_auc = self.evaluator.evaluate_model(model, x_val, y_val)

        # Resultados
        print(f'Acurácia de treino: {acc_train}')
        print(f'Acurácia de validação: {acc_val}')
        print(f'F1 Score: {f1_val}')
        print(f'Acurácia Score: {accuracy_score_val}')
        print(f'Precisão: {precision_val}')
        print(f'Recall: {recall_val}')
        print(f'AUC: {roc_auc}')

        # Predição e binarização para métricas adicionais
        y_pred = model.predict(x_val)
        y_val_bin = label_binarize(y_val, classes=[0, 1])
        y_pred_bin = label_binarize(y_pred, classes=[0, 1])

        # Plotar a matriz de confusão e o relatório de classificação
        self.evaluator.plot_confusion_matrix(y_val, y_pred)
        self.evaluator.print_classification_report(y_val, y_pred)

        # Plotar a curva ROC
        self.evaluator.plot_roc(y_val, y_pred)
