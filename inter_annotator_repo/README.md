
# Inter-Annotator Agreement EDA (Modularized Version)

This repository contains a fully modularized Exploratory Data Analysis (EDA) pipeline for analyzing inter-annotator agreement in emotion annotation datasets.

## Repository Structure

- `main.py` — Main script that executes the full EDA pipeline.
- `src/` — Folder containing all modularized Python classes:
  - `loader.py` — Load and preprocess Excel annotation files.
  - `statistics.py` — Compute descriptive statistics for annotations.
  - `plotter.py` — Generate emotion distribution plots.
  - `agreement.py` — Compute pairwise inter-annotator Cohen’s Kappa scores.
  - `confusion.py` — Generate confusion matrix heatmaps.
  - `kappa_plots.py` — Generate Kappa agreement heatmaps.
- `annotations/` — Folder containing the raw Excel annotation files (not uploaded to GitHub).
- `figures/` — Auto-generated plots saved during EDA.
- `requirements.txt` — Required Python packages.
- `.gitignore` — Files and folders excluded from version control.

## Input File Format

The Excel annotation files must follow the format:

id, text, anger, fear, joy, sadness, surprise

The code automatically renames columns and processes missing values.

## How to Run

### 1️⃣ Prepare your dataset

- Place all annotation Excel files inside the `annotations/` folder.

### 2️⃣ Install dependencies

pip install -r requirements.txt

### 3️⃣ Execute the pipeline

python main.py

- All generated plots will be saved automatically inside `figures/`.

## Output

- Descriptive statistics printed for each annotation file.
- Emotion distribution barplots.
- Cohen’s Kappa heatmaps.
- Confusion matrix heatmaps.

## Notes

- Fully modularized for reproducibility and scalability.
- Clean separation of code responsibilities using Python classes.
- Ready for thesis submission.

---

This repository is fully reproducible and thesis-ready.
