import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class KappaPlotter:
    """Create and save heatmaps for inter-annotator Kappa agreements."""

    def __init__(self, figures_dir):
        self.figures_dir = figures_dir

    def plot_kappa_heatmap(self, kappa_matrix, emotion, annotators):
        """Plot Kappa upper triangular heatmap."""
        mask = np.tril(np.ones_like(kappa_matrix, dtype=bool))
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            kappa_matrix, mask=mask, annot=True, cmap="coolwarm",
            fmt=".2f", xticklabels=annotators, yticklabels=annotators
        )
        plt.title(f"Kappa Heatmap for {emotion}")
        plt.savefig(self.figures_dir / f"{emotion}_kappa_heatmap.png")
        plt.close()
