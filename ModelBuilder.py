from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import StandardScaler


class ModelBuilder:
    def __init__(self, model_type="DecisionTree", max_depth=10):
        if model_type == "DecisionTree":
            self.model = DecisionTreeClassifier(max_depth=max_depth)
        elif model_type == "RandomForest":
            self.model = RandomForestClassifier(max_depth=max_depth)
        else:
            raise ValueError("Unsupported model type")
        self.skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)

    def build_pipeline(self, balance_strategy="oversample"):
        if balance_strategy == "oversample":
            return imbpipeline(
                [
                    ("scaler", StandardScaler()),
                    ("oversample", SMOTE()),
                    ("model", self.model),
                ]
            )
        elif balance_strategy == "undersample":
            return imbpipeline(
                [
                    ("scaler", StandardScaler()),
                    ("undersample", NearMiss(version=3)),
                    ("model", self.model),
                ]
            )
        else:
            raise ValueError("Unsupported balance strategy")

    def train_model(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return self.model

    def cross_validate_model(self, x, y):
        scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        return cross_validate(self.model, x, y, cv=self.skf, scoring=scoring)

    def get_model(self):
        return self.model
