# Main Models Repository (Supervised & Transformer Models)

This repository contains the full pipeline for supervised machine learning models and transformer-based models applied on multi-label emotion classification.

## Repository Structure

- `supervised_models.py` : Traditional ML models using TF-IDF, Word2Vec, Optuna hyperparameter tuning, RandomForest, KNN.
- `transformer_models.py` : Transformer-based embeddings (BERT, RoBERTa, XLM-RoBERTa, ModernBERT) with classical ML classifiers.
- `eng.xlsx` : Raw annotation dataset (NOT uploaded to GitHub).
- `results/` : Saved evaluation reports and intermediate outputs.
- `figures/` : Auto-generated visualizations.
- `saved_models/` : Saved trained models.
- `requirements.txt` : Required Python packages.
- `.gitignore` : Files/folders excluded from version control.

## Emotion Labels

The models classify texts into the following emotion categories:

- Anger
- Fear
- Joy
- Sadness
- Surprise

## How to Run

### Prepare your data

- Place `eng.xlsx` directly in the root of `main_models_repo/`.

The Excel file should contain:

text, anger, fear, joy, sadness, surprise

### Install dependencies

pip install -r requirements.txt

Example packages included:

pandas
numpy
scikit-learn
matplotlib
tqdm
transformers
optuna
gensim
xgboost
catboost
lightgbm
joblib
nltk

### Execute scripts

#### Traditional ML models:

python supervised_models.py

#### Transformer models:

python transformer_models.py

- Results will be saved into `results/` and `saved_models/`.
- Figures will be saved into `figures/`.

## Notes

- This repository performs **multi-label classification**.
- The Transformers extract embeddings from pretrained models and apply classical ML classifiers.
- The dataset file `eng.xlsx`, intermediate models and embeddings are excluded from GitHub for privacy.

---

This repository is fully reproducible and thesis-ready.
