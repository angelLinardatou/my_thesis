import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_kappa_heatmap(
    kappa_matrix: np.ndarray,
    emotion: str,
    annotators: list[str],
    figures_dir: Path
) -> None:
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
