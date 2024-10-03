import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
    
    def load_data(self):
        self.data = pd.read_csv(self.filepath)
        return self.data

    def split_data(self, x, y, test_size=0.15, val_size=0.15, random_state=5):
        
        x, x_teste, y, y_teste = train_test_split(
            x, y, test_size=test_size,
            stratify=y, random_state=random_state
            )
        
        x_treino, x_val, y_treino, y_val = train_test_split(
            x, y, test_size=val_size, 
            stratify=y, random_state=random_state
            )
        
        return x_treino, x_val, x_teste, y_treino, y_val, y_teste
