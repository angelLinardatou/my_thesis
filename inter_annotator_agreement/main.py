
from src.loader import load_annotation_files
from src.statistics import compute_annotation_statistics
from src.plot import plot_emotion_distribution
from src.agreement import AgreementCalculator
from src.kappa_plots import plot_kappa_heatmap
from pathlib import Path

# Define project paths
base_dir = Path(__file__).parent
annotation_dir = base_dir / "annotations"
figures_dir = Path(__file__).parent / "figures"
figures_dir.mkdir(exist_ok=True)

# Load data
dataframes = load_annotation_files(annotation_dir)

# Compute descriptive stats
statistics = compute_annotation_statistics(dataframes)

for file, stat in statistics.items():
    print(f"Statistics for {file}:\n{stat}\n")

# Plot basic distributions
for filename, df in dataframes.items():
    plot_emotion_distribution(df, filename, figures_dir)

# Compute agreement and plot heatmaps
for filename, df in dataframes.items():
    annotations = df[['anger', 'fear', 'joy', 'sadness', 'surprise']]
    for emotion in annotations.columns:

        agreement = AgreementCalculator(annotations, emotion)
        kappa_matrix = agreement.compute_kappa_matrix()

        plot_kappa_heatmap(
            kappa_matrix,
            emotion,
            annotators=[f"Annotator {i+1}" for i in range(kappa_matrix.shape[0])],
            figures_dir=figures_dir
        )


print("Full EDA pipeline completed successfully!")

