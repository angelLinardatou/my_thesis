# Thesis: Multilingual Sentiment and Emotion Analysis

This repository hosts the code and data used for my undergraduate thesis. It is organized into three main sub-projects, each inside a separate folder.

---

##  Repository Structure

### 1. `greek_training_models/`
This folder contains scripts and experiments for **fine-tuning XLM-RoBERTa** on Greek datasets for:
- 3-class sentiment analysis
- 4-class sentiment including narrator detection
- 9-class emotion classification

🔹 Datasets are stored in `greek_training_models/data/`.

🔹 Training scripts perform tokenization, fine-tuning, evaluation, and save models in `results/`.

### 2. `inter_annotator_agreement/`
This folder analyzes **inter-annotator agreement** using various statistical and visual methods.
 🔹 Datasets: stored in `annotations/`.
🔹 Main script: `main.py`, which loads annotations, visualizes distributions, and calculates agreement metrics (e.g. Fleiss’ kappa, mutual information).

### 3. `main_models/`
This folder contains additional **supervised and transformer-based models** trained on the annotation data.

🔹 Core files:
- `supervised_models.py`: trains traditional ML classifiers (e.g., SVM, Random Forest)
- `transformer_models.py`: fine-tunes transformer-based models like BERT

🔹 Dataset: `eng.csv` stored in `data/`.
