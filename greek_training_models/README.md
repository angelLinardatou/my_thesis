
# Greek Emotion & Sentiment Classification (Modularized Version)

This folder contains a fully modularized pipeline for fine-tuning transformer models (RoBERTa/XLM-RoBERTa) on Greek emotion and sentiment classification tasks.

## Repository Structure

- `main.py` — Main script that runs the full training pipeline for all datasets.
- `src/` — Folder containing all modularized Python classes:
  - `data_loader.py` — Load and preprocess the Excel datasets.
  - `tokenizer_dataset.py` — Create tokenized HuggingFace datasets.
  - `trainer.py` — Handle fine-tuning of transformer models.
  - `evaluator.py` — Evaluate trained models and generate classification reports.
- `data/` - Folder containing all the datasets
  - `gr.csv` — 3-class sentiment dataset.
  - `ib1_sentiment_probs.cvs` — 4-class sentiment dataset.
  - `ground_truth.csv` — 9-class emotion dataset.
- `requirements.txt` — Required Python packages.
- `.gitignore` — Files and folders excluded from version control.

## Tasks & Labels

| Dataset | Classes | Model |
|---------|---------|--------|
| `gr.xlsx` | negative, neutral, positive | xlm-roberta-large |
| `ib1_sentiment_probs.xlsx` | negative, neutral, positive, narrator | xlm-roberta-large |
| `ground_truth.xlsx` | 9 emotions | xlm-roberta-base |

## How to Run

### 1️. Prepare your data

Place the three input files in the root directory:

- `gr.xlsx`
- `ib1_sentiment_probs.xlsx`
- `ground_truth.xlsx`

### 2️. Install dependencies

```bash
pip install -r requirements.txt
```

### 3️. Execute the script

```bash
python main.py
```

## Output

- Trained models will be saved inside the `results/` folder.
- Evaluation reports are printed automatically after training.

## Notes

- Fully modularized and reproducible.
- Clean separation into classes for data loading, training, evaluation.
- Handles multiple tasks in a single run.
- Ready for thesis submission.

---

This repository is fully reproducible and thesis-ready.
