from DataLoader import DataLoader
from ModelBuilder import ModelBuilder
from ModelEvaluator import ModelEvaluator
from Trainer import Trainer


# Instanciando as classes
data_loader = DataLoader(filepath='data/emp_automovel.csv')
model_builder = ModelBuilder(max_depth=5) #, min_samples_split=5, min_samples_leaf=5
evaluator = ModelEvaluator()
trainer = Trainer(data_loader, model_builder, evaluator)

# Treinando e avaliando o modelo
trainer.train_and_evaluate()
