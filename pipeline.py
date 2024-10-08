from DataLoader import DataLoader
from ModelBuilder import ModelBuilder
from ModelEvaluator import ModelEvaluator
from Trainer import Trainer
from TrainerWorkflow import TrainerWorkflow
from ModelSaver import ModelSaver

def run_pipeline(filepath, model_type="DecisionTree", balance_strategy="oversample", save_model=False):
    # Carregar os dados
    data_loader = DataLoader(filepath=filepath)
    
    # Definir o tipo de modelo e balanceamento
    model_builder = ModelBuilder(model_type=model_type)
    
    # Avaliador para gerar métricas e plots
    evaluator = ModelEvaluator()
    
    # Criar workflow de treinamento
    trainer = Trainer(model_builder)
    workflow = TrainerWorkflow(data_loader, trainer, evaluator)
    
    # Executar o pipeline de treinamento e avaliação
    workflow.train_and_evaluate()
    
    # Salvar o modelo se necessário
    if save_model:
        model_saver = ModelSaver()
        model_saver.save_model(model_builder.get_model(), model_type=model_type, balance_strategy=balance_strategy)


if __name__ == "__main__":
    # Executar o pipeline com o caminho do arquivo de dados
    filepath = 'data/emp_automovel.csv'  # O caminho do arquivo de dados
    run_pipeline(filepath, model_type="DecisionTree", balance_strategy="undersample", save_model=True)
