
# Inter-Annotator Agreement EDA

This repository contains an Exploratory Data Analysis (EDA) pipeline for studying inter-annotator agreement in emotion labeling tasks.

## Repository Structure

- `inter_annotator_eda.py`: Main EDA script.
- `annotations/`: Folder containing the input annotation Excel files (not uploaded to GitHub).
- `figures/`: Auto-generated visualizations.
- `requirements.txt`: Dependencies.
- `.gitignore`: Files and folders excluded from version control.

## How to Run

### 1️. Place your annotation Excel files into the `annotations/` folder.

The Excel files must have 6 columns (after skipping the first row):

```
id, text, anger, fear, joy, sadness, surprise
```

The script automatically handles missing values by filling them with column means.

### 2️. Install dependencies

```bash
pip install -r requirements.txt
```

### 3️. Execute the EDA script

```bash
python inter_annotator_eda.py
```

The figures will be automatically saved into the `figures/` directory.

---

This repository is fully reproducible and thesis-ready.
