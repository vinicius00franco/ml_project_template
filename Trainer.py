from sklearn.preprocessing import label_binarize

# Classe Trainer - Responsável pelo treinamento do modelo
class Trainer:
    def __init__(self, model_builder):
        self.model_builder = model_builder

    def train_model(self, x_train, y_train):
        return self.model_builder.train_model(x_train, y_train)

    def cross_validate_model(self, x, y):
        return self.model_builder.cross_validate_model(x, y)
    
    def cross_validate_recall_model(self, x, y ):
        return self.model_builder.cross_validate_recall_model(x, y)
