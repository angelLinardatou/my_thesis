import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class ModelEvaluator:
    """Generate classification reports and confusion matrix for predictions."""

    def __init__(self, label_names):
        self.label_names = label_names

    def evaluate(self, y_true, y_pred):
        """Print classification report."""
        report = classification_report(y_true, y_pred, target_names=self.label_names)
        print(report)
        return report

    def compute_confusion_matrix(self, y_true, y_pred):
        """Compute confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        return cm
 
