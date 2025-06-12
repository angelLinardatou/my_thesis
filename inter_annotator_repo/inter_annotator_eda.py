import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ===========================================
# Setup directories
# ===========================================

# Set base directory to the repository root
base_dir = Path(__file__).resolve().parent
annotation_dir = base_dir / 'annotations'
figure_dir = base_dir / 'figures'
figure_dir.mkdir(exist_ok=True)

# ===========================================
# Load annotation files
# ===========================================

xlsx_files = list(annotation_dir.glob('*.xlsx'))
dataframes = {f.name: pd.read_excel(f, skiprows=1) for f in xlsx_files}

# Rename columns for consistency
for fname, df in dataframes.items():
    df.columns = ['id', 'text', 'anger', 'fear', 'joy', 'sadness', 'surprise']

# ===========================================
# Handle missing values (fill with column mean)
# ===========================================

for file_name, df in dataframes.items():
    for column in ['anger', 'fear', 'joy', 'sadness', 'surprise']:
        mean_value = df[column].mean()
        df[column].fillna(mean_value, inplace=True)

# ===========================================
# Basic statistics for emotion columns
# ===========================================

for file_name, df in dataframes.items():
    print(f'File: {file_name}')
    print(df[['anger', 'fear', 'joy', 'sadness', 'surprise']].describe())
    print("-"*50)

# ===========================================
# Emotion distribution per annotator
# ===========================================

for index, (file_name, df) in enumerate(dataframes.items(), start=1):
    annotator_label = f'Annotator {index}'
    emotion_counts = df[['anger', 'fear', 'joy', 'sadness', 'surprise']].sum()

    plt.figure(figsize=(10, 6))
    emotion_counts.plot(kind='bar', color=['red', 'blue', 'green', 'yellow', 'purple'])
    plt.title(f'Emotion Distribution - {annotator_label}')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(figure_dir / f'plot_{index:02d}.png')
    plt.close()

# ===========================================
# Total emotion counts across annotators
# ===========================================

total_emotion_counts = sum(df[['anger', 'fear', 'joy', 'sadness', 'surprise']].sum() for df in dataframes.values())

plt.figure(figsize=(10, 6))
total_emotion_counts.plot(kind='bar', color=['red', 'blue', 'green', 'yellow', 'purple'])
plt.title('Total Emotion Distribution Across All Annotators')
plt.xlabel('Emotion')
plt.ylabel('Total Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(figure_dir / 'plot_total.png')
plt.close()

# ===========================================
# Emotion counts per annotator per emotion
# ===========================================

emotions = ['anger', 'fear', 'joy', 'sadness', 'surprise']
emotion_counts_per_file = {emotion: [df[emotion].sum() for df in dataframes.values()] for emotion in emotions}

for emotion, counts in emotion_counts_per_file.items():
    plt.figure(figsize=(8, 5))
    annotator_labels = [f'Annotator {i+1}' for i in range(len(dataframes))]
    plt.bar(annotator_labels, counts, color='skyblue')
    plt.title(f'{emotion.capitalize()} Distribution Across Annotators')
    plt.xlabel('Annotators')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(figure_dir / f'plot_{emotion}.png')
    plt.close()

# ===========================================
# Correlation heatmap between annotators
# ===========================================

emotion_data = {
    file_name: df[['anger', 'fear', 'joy', 'sadness', 'surprise']].sum()
    for file_name, df in dataframes.items()
}

emotion_df = pd.DataFrame(emotion_data).T
emotion_df.index.name = "Files"
correlation_matrix = emotion_df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title("Upper Triangular Heatmap of Correlation Between Annotators")
plt.tight_layout()
plt.savefig(figure_dir / 'plot_correlation.png')
plt.close()

print("EDA analysis completed successfully and all figures saved.")
