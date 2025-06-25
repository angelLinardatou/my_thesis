import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path


def plot_confusion_matrix(

    y_true: list,
    y_pred: list,
    labels: list,
    filename: str,
    figures_dir: Path
) -> None:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=labels, yticklabels=labels)
    plt.title("Overall Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True Label")
    plt.savefig(figures_dir / filename)
    plt.close()
