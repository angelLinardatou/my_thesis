
# Greek Emotion Classification Models

This repository contains emotion classification models trained on Greek language datasets, including fine-tuning of transformer-based models.

## Repository Structure

- `greek_roberta_models.py`: Main training and evaluation script for Greek transformer models.
- `gr.xlsx`: Main dataset with text and labels (not uploaded to GitHub).
- `ground_truth.xlsx`: Annotated ground truth labels (not uploaded to GitHub).
- `ib1_sentiment_probs.xlsx`: Additional sentiment probabilities (not uploaded to GitHub).
- `figures/`: Auto-generated visualizations.
- `results/`: Saved evaluation reports.
- `requirements.txt`: Required Python packages.
- `.gitignore`: Files/folders excluded from version control.

## Emotion Labels

The models classify Greek texts into:

- Positive
- Negative
- Neutral
- Narrator (if used)

## How to Run

### 1️. Prepare your data

Place the following files into the root directory:

- `gr.xlsx`
- `ground_truth.xlsx`
- `ib1_sentiment_probs.xlsx`

### 2️. Install dependencies

```bash
pip install -r requirements.txt
```

### 3️. Execute the script

```bash
python greek_roberta_models.py
```

The results will be saved into the `results/` and `figures/` folders.

---

This repository is fully reproducible and thesis-ready.
