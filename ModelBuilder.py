from sklearn.tree import DecisionTreeClassifier

class ModelBuilder:
    def __init__(self, max_depth=5):#, min_samples_split=5, min_samples_leaf=5
        self.model = DecisionTreeClassifier(max_depth=max_depth) #, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf

    def train_model(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return self.model

    def get_model(self):
        return self.model
