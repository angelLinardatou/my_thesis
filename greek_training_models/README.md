# Greek Training Models (RoBERTa)

This repository contains code for training and evaluating multilingual models on Greek-language sentiment and emotion datasets.

## Model

- **XLM-RoBERTa** was fine-tuned using Greek datasets.
- The task includes classification into binary, trinary, and multi-emotion labels.
- Evaluation results and figures are automatically generated and saved.

## Structure

```
greek_training_models/
├── greek_roberta_models.py      # Fine-tuned XLM-RoBERTa model code
├── results/
│   └── metrics.csv              # Evaluation scores
├── figures/
│   ├── plot_01_sentiment_distribution.png
│   └── plot_02_label_distribution.png
```

## Usage

```bash
python greek_roberta_models.py
```

All plots will be saved in the `figures/` folder, and performance metrics in `results/`.