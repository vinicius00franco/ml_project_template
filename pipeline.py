from DataLoader import DataLoader
from ModelBuilder import ModelBuilder
from ModelEvaluator import ModelEvaluator
from TrainerWorkflow import TrainerWorkflow
from Trainer import Trainer



# Instanciando as classes
data_loader = DataLoader(filepath='data/emp_automovel.csv')
model_builder = ModelBuilder(max_depth=5) #, min_samples_split=5, min_samples_leaf=5
evaluator = ModelEvaluator()


trainer = Trainer(model_builder)  # model_builder já deve ser instanciado previamente
workflow = TrainerWorkflow(data_loader, trainer, evaluator)  # data_loader e evaluator também já devem ser instanciados
workflow.train_and_evaluate()

