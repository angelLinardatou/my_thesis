import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    label_names: list[str]
) -> str:
        """Print classification report."""
        report = classification_report(y_true, y_pred, target_names=label_names)
        print(report)
        return report

def compute_confusion_matrix(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray
) -> np.ndarray:
        """Compute confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        return cm
 
