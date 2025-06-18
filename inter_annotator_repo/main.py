# Plot emotion distribution for each file
import numpy as np
import pandas as pd
import sys
import pathlib 

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from src.loader import AnnotationLoader
from src.statistics import AnnotationStatistics
from src.plot import AnnotationPlotter
from src.agreement import AgreementCalculator
from src.confusion import ConfusionMatrixPlotter
from src.kappa_plots import KappaPlotter
from pathlib import Path

# Define project paths
base_dir = Path(__file__).parent
annotation_dir = base_dir / "annotations"
figures_dir = Path(__file__).parent / "figures"
figures_dir.mkdir(exist_ok=True)

# Load data
loader = AnnotationLoader(annotation_dir)
dataframes = loader.load_files()

# Compute descriptive stats
stats = AnnotationStatistics(dataframes)
statistics = stats.compute_statistics()

for file, stat in statistics.items():
    print(f"Statistics for {file}:\n{stat}\n")

# Plot basic distributions
plotter = AnnotationPlotter(figures_dir)
for filename, df in dataframes.items():
    plotter.plot_emotion_distribution(df, filename)

# Example agreement calculation for one file:
# You can loop over files or do per emotion analysis here
# Just as an example:
for filename, df in dataframes.items():
    annotations = df[['anger', 'fear', 'joy', 'sadness', 'surprise']]
    for emotion in annotations.columns:
        # In your real case you'll load the full annotation matrix from your big files

        agreement = AgreementCalculator(annotations, emotion)
        kappa_matrix = agreement.compute_kappa_matrix()

        kappa_plotter = KappaPlotter(figures_dir)

        kappa_plotter.plot_kappa_heatmap(kappa_matrix, emotion, annotators=[f"Annotator {i+1}" for i in range(kappa_matrix.shape[0])])

print("Full EDA pipeline completed successfully!")

