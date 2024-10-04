from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold

class ModelBuilder:
    def __init__(self, max_depth=5):#, min_samples_split=5, min_samples_leaf=5
        self.model = DecisionTreeClassifier(max_depth=max_depth) #, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf
        self.kf = KFold(n_splits = 5, shuffle = True, random_state = 5)
        self.skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 5)

    def train_model(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return self.model
    
    def cross_validate_model(self, x, y ):
        cv_resultados = cross_validate(self.model, x, y, cv = self.kf)
        return cv_resultados
    
    def cross_validate_recall_model(self, x, y ):
        cv_resultados = cross_validate(self.model, x, y, cv = self.skf, scoring='recall')
        return cv_resultados


    def get_model(self):
        return self.model
    
    @staticmethod
    def evaluate_analysis(data):
        data['inadimplente'].value_counts(normalize = True)
