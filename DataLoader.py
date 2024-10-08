import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

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

    def oversample_data(self, x, y):
        smote = SMOTE()
        return smote.fit_resample(x, y)

    def undersample_data(self, x, y):
        undersample = NearMiss(version=3)
        return undersample.fit_resample(x, y)
        