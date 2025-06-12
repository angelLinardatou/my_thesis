# Greek Sentiment and Emotion Fine-Tuning (Greek Language)

This repository contains fine-tuning experiments on **Greek language** sentiment and emotion classification using the XLM-RoBERTa model. The dataset includes three different tasks with varying label schemes.

## Repository Structure

- `greek_roberta_models.py` : Main training and evaluation script.
- `gr.xlsx` : Dataset for 3-class sentiment classification (negative, neutral, positive).
- `ib1_sentiment_probs.xlsx` : Dataset for 4-class sentiment classification (negative, neutral, positive, narrator).
- `ground_truth.xlsx` : Dataset for 9-class emotion classification.
- `figures/` : Auto-generated plots.
- `results/` : Saved evaluation results.
- `requirements.txt` : Required Python packages.
- `.gitignore` : Files and folders excluded from version control.

## How to Run

 **1. Prepare Data**

- Place the datasets (`gr.xlsx`, `ib1_sentiment_probs.xlsx`, `ground_truth.xlsx`) inside the main repository folder.

**2. Install Dependencies**

```bash
pip install -r requirements.txt
