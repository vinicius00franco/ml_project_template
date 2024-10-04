from sklearn.preprocessing import label_binarize

# Classe Trainer - Responsável pelo treinamento do modelo
class Trainer:
    def __init__(self, model_builder):
        self.model_builder = model_builder

    def train_model(self, x_train, y_train):
        return self.model_builder.train_model(x_train, y_train)
    
    
    def train_model_undersample(self, x_balanceado, y_balanceado):
        return self.model_builder.train_model_undersample(x_balanceado, y_balanceado)
    

    def cross_validate_model(self, x, y):
        return self.model_builder.cross_validate_model(x, y)
    
    def cross_validate_recall_model(self, x, y ):
        return self.model_builder.cross_validate_recall_model(x, y)
    
    def cross_validate_recall_model_balanceado(self, x, y  ):
        return self.model_builder.cross_validate_recall_model_balanceado(x, y)

   
    def cross_validate_recall_model_balanceado_undersample(self, x, y):
        return self.model_builder.cross_validate_recall_model_balanceado_undersample(x, y)
