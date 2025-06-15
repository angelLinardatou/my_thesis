
# Main Models Repository (Supervised & Transformer Models - Full Modularized Version)

This repository contains the full modularized pipeline for both supervised machine learning models and transformer-based models applied to multi-label emotion classification on English text data.

## Repository Structure

- `main.py` — Full supervised ML pipeline (TF-IDF, Word2Vec).
- `transformer_main.py` — Full transformer embeddings ML pipeline.
- `src/` — Folder containing all modularized code:
  - `data_loader.py` — Load and preprocess the Excel dataset.
  - `text_cleaner.py` — Clean text (URLs, symbols, lowercasing).
  - `features_tfidf.py` — Extract TF-IDF features.
  - `features_word2vec.py` — Extract Word2Vec features.
  - `supervised_trainer.py` — Train ML models on TF-IDF & Word2Vec features.
  - `transformer_embedding_extractor.py` — Extract embeddings from transformer models.
  - `transformer_trainer.py` — Train ML models on transformer embeddings.
  - `evaluation.py` — Evaluate models and save classification reports.
- `eng.xlsx` — Multi-label emotion annotation dataset.
- `results/` — Saved supervised ML evaluation reports.
- `results_transformers/` — Saved transformer-based evaluation reports.
- `requirements.txt` — Required Python packages.

## Emotion Labels

The models classify texts into the following 5 emotion categories:

- Anger
- Fear
- Joy
- Sadness
- Surprise

## How to Run

### 1️⃣ Prepare your data

Place `eng.xlsx` directly inside the project root directory.

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Execute the pipelines

#### For supervised models:

```bash
python main.py
```

#### For transformer models:

```bash
python transformer_main.py
```

The outputs (evaluation reports) will be automatically stored in `results/` and `results_transformers/`.

## Output

- Classification reports (F1-scores, precision, recall)
- Saved models and reproducible results
- Fully modularized codebase

## Notes

- Fully modularized and reproducible.
- Clean separation between supervised ML and transformer-based pipelines.
- Thesis-ready submission structure.

---

This repository is fully reproducible and thesis-ready.
