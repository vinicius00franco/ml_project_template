import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss


class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.oversample = SMOTE()
        self.undersample = NearMiss(version=3)
    
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

    def oversample_data(self, x, y):
        x_balanceado, y_balanceado = self.oversample.fit_resample(x, y)
        balance = y_balanceado.value_counts(normalize=True)
        
        if balance.min() != balance.max():
            raise ValueError("Os dados não estão perfeitamente equilibrados.")
            
        print("Os dados estão equilibrados.")
        
        return x_balanceado, y_balanceado
    
    def undersample_data(self, x, y):
        x_balanceado, y_balanceado = self.undersample.fit_resample(x, y)
        # balance = y_balanceado.value_counts(normalize=True)
        
        # if balance.min() != balance.max():
        #     raise ValueError("Os dados não estão perfeitamente equilibrados.")
            
        # print("Os dados estão equilibrados.")
        
        return x_balanceado, y_balanceado
        