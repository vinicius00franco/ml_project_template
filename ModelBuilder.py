from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss


class ModelBuilder:
    def __init__(self, max_depth=10):#, min_samples_split=5, min_samples_leaf=5
        self.model = DecisionTreeClassifier(max_depth=max_depth) #, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf
        self.kf = KFold(n_splits = 5, shuffle = True, random_state = 5)
        self.skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 5)
        self.pipeline = imbpipeline([('oversample', SMOTE()), ('arvore', self.model)])
        self.pipeline_undersample = imbpipeline([('undersample',NearMiss(version=3)), ('arvore', self.model)])


    def train_model(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return self.model
    
    def train_model_undersample(self, x_balanceado, y_balanceado):
        self.model.fit(x_balanceado, y_balanceado)
        return self.model
    
    def cross_validate_model(self, x, y ):
        cv_resultados = cross_validate(self.model, x, y, cv = self.kf)
        return cv_resultados
    
    def cross_validate_recall_model(self, x, y ):
        cv_resultados = cross_validate(self.model, x, y, cv = self.skf, scoring='recall')
        return cv_resultados
    
    def cross_validate_recall_model_balanceado(self, x, y ):
        cv_resultados = cross_validate(self.pipeline, x, y, cv = self.skf, scoring='recall')
        return cv_resultados

    def cross_validate_recall_model_balanceado_undersample(self, x, y ):
        cv_resultados = cross_validate(self.pipeline_undersample, x, y, cv = self.skf, scoring='recall')
        return cv_resultados



    def get_model(self):
        return self.model
    
    @staticmethod
    def evaluate_analysis(data):
        data['inadimplente'].value_counts(normalize = True)
