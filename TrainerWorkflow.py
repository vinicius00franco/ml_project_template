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
        
        x_balanceado, y_balanceado = self.data_loader.undersample_data(x, y)
        
        # Treinar o modelo
        model = self.model_builder.train_model(x_treino, y_treino)
        model_undersample = self.model_builder.train_model_undersample(x_balanceado, y_balanceado)        
        y_previsto_under = model_undersample.predict(x_teste)

        
        # Cross-validation para avaliar o modelo
        cv_results = self.model_builder.cross_validate_model(x, y)
        cv_recall_results = self.model_builder.cross_validate_recall_model(x,y)
        cv_recall_results_balanceado = self.model_builder.cross_validate_recall_model_balanceado(x, y)
        cv_recall_results_balanceado_undersample = self.model_builder.cross_validate_recall_model_balanceado_undersample(x, y)

        # self.intervalo_conf(cv_results, "geral")
        self.intervalo_conf(cv_recall_results, "scoring recall")
        self.intervalo_conf(cv_recall_results_balanceado, "scoring recall balanceado")
        self.intervalo_conf(cv_recall_results_balanceado_undersample, "scoring recall balanceado under")
        

        
        # Avaliar o modelo nos dados de treino e validação
        acc_train = model.score(x_treino, y_treino)
        acc_val, f1_val, accuracy_score_val, precision_val, recall_val, roc_auc = self.evaluator.evaluate_model(model, x_val, y_val)
        

        
        

        # Resultados
        print(f'\n\n Acurácia de treino: {acc_train}\n\n')
        print(f'Acurácia de validação: {acc_val}\n\n')
        # print(f'F1 Score: {f1_val}\n\n')
        print(f'Acurácia Score: {accuracy_score_val}\n\n')
        print(f'Precisão: {precision_val}\n\n')
        print(f'Recall: {recall_val}\n\n')
        # print(f'AUC: {roc_auc}\n\n')

        # Predição e binarização para métricas adicionais
        y_pred = model.predict(x_val)
        y_val_bin = label_binarize(y_val, classes=[0, 1])
        y_pred_bin = label_binarize(y_pred, classes=[0, 1])

        # Plotar a matriz de confusão e o relatório de classificação
        self.evaluator.plot_confusion_matrix(y_val, y_pred)
        self.evaluator.print_classification_report(y_val, y_pred)
        
        # plotar matriz de teste do modelo
        self.evaluator.print_classification_report(y_teste, y_previsto_under)
        self.evaluator.print_confusion_matriz_report(y_teste, y_previsto_under)

        # Plotar a curva ROC
        self.evaluator.plot_roc(y_val, y_pred)
        
    def intervalo_conf(self,resultados, texto: str):
        media = resultados['test_score'].mean()
        desvio_padrao = resultados['test_score'].std()
        print(f'Intervalo de confiança {texto}: [{media - 2*desvio_padrao}, {min(media + 2*desvio_padrao, 1)}]\n\n')

