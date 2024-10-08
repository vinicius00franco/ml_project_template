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
        self.pipeline = None

    def build_pipeline(self, balance_strategy="oversample"):
        if balance_strategy == "oversample":
            self.pipeline = imbpipeline(
                [
                    ("scaler", StandardScaler()),
                    ("oversample", SMOTE()),
                    ("model", self.model),
                ]
            )
        elif balance_strategy == "undersample":
            self.pipeline = imbpipeline(
                [
                    ("scaler", StandardScaler()),
                    ("undersample", NearMiss(version=3)),
                    ("model", self.model),
                ]
            )
        else:
            raise ValueError("Unsupported balance strategy")
        return self.pipeline

    def train_model(self, x_train, y_train):
        # Garante que o pipeline foi criado
        if self.pipeline is None:
            raise ValueError(
                "You need to build the pipeline before training the model."
            )

        self.pipeline.fit(x_train, y_train)
        return self.pipeline

    def cross_validate_model(self, x, y):
        # Garante que o pipeline foi criado
        if self.pipeline is None:
            raise ValueError(
                "You need to build the pipeline before cross-validating the model."
            )

        scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        # Usar o pipeline em vez do modelo diretamente
        return cross_validate(self.pipeline, x, y, cv=self.skf, scoring=scoring)

    def get_model(self):
        return self.pipeline if self.pipeline else self.model
