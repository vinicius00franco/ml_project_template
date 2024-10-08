from sklearn.preprocessing import label_binarize
from ml.Trainer import Trainer


class TrainerWorkflow:
    def __init__(self, data_loader, model_builder, evaluator , balance_strategy   ):
        self.data_loader = data_loader
        self.model_builder = model_builder
        self.evaluator = evaluator
        self.balance_strategy = balance_strategy
        self.trainer = Trainer(model_builder)

    def train_and_evaluate(self):
        data = self.data_loader.load_data()
        x = data.drop("inadimplente", axis=1)
        y = data["inadimplente"]

        # Split data
        x_train, x_val, x_test, y_train, y_val, y_test = self.data_loader.split_data(
            x, y
        )

        # Garantir que o pipeline seja construído com a estratégia de balanceamento
        self.model_builder.build_pipeline(balance_strategy=self.balance_strategy)  # Garantir a construção do pipeline
        
        
        # Cross-validate the model to get performance metrics
        cross_val_results = self.trainer.cross_validate(x_train, y_train)

        # Exibir os resultados da validação cruzada
        print("Resultados da Validação Cruzada:")
        for metric, scores in cross_val_results.items():
            print(f"{metric}: {scores.mean():.4f} (+/- {scores.std():.4f})")

        # Treinar o modelo final com todos os dados de treino
        model = self.trainer.train_model(x_train, y_train)

        # Evaluate model
        evaluation_metrics = self.evaluator.evaluate_model(model, x_val, y_val)

        print(f"Accuracy: {evaluation_metrics['accuracy']:.8f}")
        print(f"F1 Score: {evaluation_metrics['f1_score']:.8f}")
        print(f"Precision: {evaluation_metrics['precision']:.8f}")
        print(f"Recall: {evaluation_metrics['recall']:.8f}")
        if evaluation_metrics["roc_auc"] is not None:
            print(f"ROC AUC: {evaluation_metrics['roc_auc']:.8f}")

        # Plotting results
        y_pred = model.predict(x_val)
        self.evaluator.plot_confusion_matrix(y_val, y_pred)
        self.evaluator.plot_roc_curve(y_val, y_pred)
