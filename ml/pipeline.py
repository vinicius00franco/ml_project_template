from ml.DataLoader import DataLoader
from ml.ModelBuilder import ModelBuilder
from ml.ModelEvaluator import ModelEvaluator
from ml.Trainer import Trainer
from ml.TrainerWorkflow import TrainerWorkflow
from ml.ModelSaver import ModelSaver


def run_pipeline(
    filepath, model_type="DecisionTree", balance_strategy="oversample", save_model=False
):
    # Carregar os dados
    data_loader = DataLoader(filepath=filepath)

    # Definir o tipo de modelo e balanceamento
    model_builder = ModelBuilder(model_type=model_type)

    # Avaliador para gerar métricas e plots
    evaluator = ModelEvaluator()

    # Criar workflow de treinamento com a estratégia de balanceamento
    workflow = TrainerWorkflow(data_loader, model_builder, evaluator, balance_strategy)

    # Executar o pipeline de treinamento e avaliação
    workflow.train_and_evaluate()

    # Salvar o modelo se necessário
    if save_model:
        model_saver = ModelSaver()
        model_saver.save_model(
            model_builder.get_model(),
            model_type=model_type,
            balance_strategy=balance_strategy,
        )


if __name__ == "__main__":
    # Executar o pipeline com o caminho do arquivo de dados
    filepath = "data/emp_automovel.csv"  # O caminho do arquivo de dados
    # Rodar o pipeline com balanceamento por undersample (NearMiss)
    run_pipeline(
        filepath="data/emp_automovel.csv",
        model_type="DecisionTree",
        balance_strategy="undersample",
        save_model=True,
    )
    run_pipeline(
        filepath="data/emp_automovel.csv",
        model_type="DecisionTree",
        balance_strategy="oversample",
        save_model=True,
    )

    # Rodar o pipeline com balanceamento por oversample (SMOTE)
    run_pipeline(
        filepath="data/emp_automovel.csv",
        model_type="RandomForest",
        balance_strategy="undersample",
        save_model=False,
    )
    run_pipeline(
        filepath="data/emp_automovel.csv",
        model_type="RandomForest",
        balance_strategy="oversample",
        save_model=False,
    )
