import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class ConfusionMatrixPlotter:
    """Create and save confusion matrix heatmaps."""

    def __init__(self, figures_dir):
        self.figures_dir = figures_dir

    def plot_confusion_matrix(self, y_true, y_pred, labels, filename):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
        plt.title("Overall Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True Label")
        plt.savefig(self.figures_dir / filename)
        plt.close()
