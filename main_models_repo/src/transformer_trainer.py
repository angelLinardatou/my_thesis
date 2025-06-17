import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier

class TransformerTrainer:
    """Train classical ML models on transformer embeddings."""

    def __init__(self):
        self.models = {}

    def train_logistic_regression(self, X_train, Y_train):
        """Train Logistic Regression."""
        clf = MultiOutputClassifier(LogisticRegression(max_iter=1000))
        clf.fit(X_train, Y_train)
        self.models['LogisticRegression'] = clf

    def train_random_forest(self, X_train, Y_train):
        """Train Random Forest."""
        clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
        clf.fit(X_train, Y_train)
        self.models['RandomForest'] = clf

    def train_svm(self, X_train, Y_train):
        """Train Support Vector Machine."""
        clf = MultiOutputClassifier(SVC(probability=True))
        clf.fit(X_train, Y_train)
        self.models['SVM'] = clf

    def predict(self, model_name, X_test):
        """Predict using selected trained model."""
        model = self.models.get(model_name)
        if model:
            return model.predict(X_test)
        else:
            raise ValueError(f"Model {model_name} not trained yet.")
 
