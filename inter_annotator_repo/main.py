from inter_annotator_eda import AnnotationStatistics, AnnotationPlotter, AgreementCalculator, KappaPlotter

# Set figures output directory
figures_dir = "figures"

# Load all dataframes (προτείνεται να έχει γίνει νωρίτερα)
# dataframes = {...}  <-- Ensure this dict is loaded correctly

# Compute statistics
stats = AnnotationStatistics(dataframes)
statistics = stats.compute_statistics()

# Print stats per file
for file, stat in statistics.items():
    print(f"Statistics for {file}:")
    print(stat)
    print("\n")

# Plot emotion distribution for each file
plotter = AnnotationPlotter(figures_dir)
for filename, df in dataframes.items():
    plotter.plot_emotion_distribution(df, filename)

# Agreement calculation and Kappa heatmaps
for filename, df in dataframes.items():
    annotations = df[['anger', 'fear', 'joy', 'sadness', 'surprise']]
    for emotion in annotations.columns:
        agreement = AgreementCalculator(annotations, emotion)
        kappa_matrix = agreement.compute_kappa_matrix()

        kappa_plotter = KappaPlotter(figures_dir)
        kappa_plotter.plot_kappa_heatmap(
            kappa_matrix,
            emotion,
            annotators=[f"Annotator {i+1}" for i in range(kappa_matrix.shape[0])],
            filename_prefix=f"{filename}_{emotion}"
        )

print("Full EDA pipeline completed successfully!")from pathlib import Path
import numpy as np
import pandas as pd

from src.loader import AnnotationLoader
from src.statistics import AnnotationStatistics
from src.plotter import AnnotationPlotter
from src.agreement import AgreementCalculator
from src.confusion import ConfusionMatrixPlotter
from src.kappa_plots import KappaPlotter

# Define project paths
base_dir = Path(__file__).parent
annotation_dir = base_dir / "annotations"
figures_dir = base_dir / "figures"

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
