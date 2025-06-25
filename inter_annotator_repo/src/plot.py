import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def plot_emotion_distribution(df, filename, figure_dir)(
    df: pd.DataFrame,
    filename: str,
    figures_dir: Path
) -> None:
        """Plot emotion label distribution for one file."""
        df[['anger', 'fear', 'joy', 'sadness', 'surprise']].sum().plot(kind='bar')
        plt.title(f"Emotion Distribution: {filename}")
        plt.ylabel("Count")
        plt.savefig(self.figures_dir / f"{filename}_distribution.png")
        plt.close()
