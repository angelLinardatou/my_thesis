import matplotlib.pyplot as plt

class AnnotationPlotter:
    """Create and save plots for annotation data."""

    def __init__(self, figures_dir):
        self.figures_dir = figures_dir

    def plot_emotion_distribution(self, df, filename):
        """Plot emotion label distribution for one file."""
        df[['anger', 'fear', 'joy', 'sadness', 'surprise']].sum().plot(kind='bar')
        plt.title(f"Emotion Distribution: {filename}")
        plt.ylabel("Count")
        plt.savefig(self.figures_dir / f"{filename}_distribution.png")
        plt.close()
