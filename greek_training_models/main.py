from pathlib import Path
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.data_loader import DataLoaderManager
from src.tokenizer_dataset import CustomDataset
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator

# Paths
base_dir = Path(__file__).parent
data_dir = base_dir
results_dir = base_dir / "results"
results_dir.mkdir(exist_ok=True)

# Initialize loader
data_loader = DataLoaderManager(data_dir)

##########################
# TASK 1 — gr.xlsx (3-class sentiment)
##########################

# Load data
gr_df = data_loader.load_dataset("gr.csv")

# Mapping
gr_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
gr_df = data_loader.map_labels(gr_df, 'label', 'label_num', gr_mapping)

# Split
gr_train_texts, gr_val_texts, gr_train_labels, gr_val_labels = train_test_split(
    gr_df['text'], gr_df['label_num'], test_size=0.2, random_state=42
)

# Tokenization
model_name_gr = "xlm-roberta-large"
tokenizer_gr = AutoTokenizer.from_pretrained(model_name_gr)
train_encodings_gr = tokenizer_gr(list(gr_train_texts), truncation=True, padding=True)
val_encodings_gr = tokenizer_gr(list(gr_val_texts), truncation=True, padding=True)

# Dataset
train_dataset_gr = CustomDataset(train_encodings_gr, list(gr_train_labels))
val_dataset_gr = CustomDataset(val_encodings_gr, list(gr_val_labels))

# Model
model_gr = AutoModelForSequenceClassification.from_pretrained(model_name_gr, num_labels=3)

# Train
trainer_gr = ModelTrainer(model_gr, train_dataset_gr, val_dataset_gr, results_dir / "gr", epochs=3, batch_size=8)
trainer_gr.train()
trainer_gr.save_model(results_dir / "gr_model")

# Evaluation
model_gr.eval()
val_preds_gr = trainer_gr.trainer.predict(val_dataset_gr)
pred_labels_gr = torch.argmax(torch.tensor(val_preds_gr.predictions), axis=1).numpy()

evaluator_gr = ModelEvaluator(["negative", "neutral", "positive"])
evaluator_gr.evaluate(gr_val_labels, pred_labels_gr)

##########################
# TASK 2 — ib1_sentiment_probs.xlsx (4-class sentiment)
##########################

ib1_df = data_loader.load_dataset("ib1_sentiment_probs.csv")
ib1_mapping = {'negative': 0, 'neutral': 1, 'positive': 2, 'narrator': 3}
ib1_df = data_loader.map_labels(ib1_df, 'final_sentiment', 'label_num', ib1_mapping)

ib1_train_texts, ib1_val_texts, ib1_train_labels, ib1_val_labels = train_test_split(
    ib1_df['text'], ib1_df['label_num'], test_size=0.2, random_state=42
)

model_name_ib1 = "xlm-roberta-large"
tokenizer_ib1 = AutoTokenizer.from_pretrained(model_name_ib1)
train_encodings_ib1 = tokenizer_ib1(list(ib1_train_texts), truncation=True, padding=True)
val_encodings_ib1 = tokenizer_ib1(list(ib1_val_texts), truncation=True, padding=True)

train_dataset_ib1 = CustomDataset(train_encodings_ib1, list(ib1_train_labels))
val_dataset_ib1 = CustomDataset(val_encodings_ib1, list(ib1_val_labels))

model_ib1 = AutoModelForSequenceClassification.from_pretrained(model_name_ib1, num_labels=4)

trainer_ib1 = ModelTrainer(model_ib1, train_dataset_ib1, val_dataset_ib1, results_dir / "ib1", epochs=3, batch_size=8)
trainer_ib1.train()
trainer_ib1.save_model(results_dir / "ib1_model")

model_ib1.eval()
val_preds_ib1 = trainer_ib1.trainer.predict(val_dataset_ib1)
pred_labels_ib1 = torch.argmax(torch.tensor(val_preds_ib1.predictions), axis=1).numpy()

evaluator_ib1 = ModelEvaluator(["negative", "neutral", "positive", "narrator"])
evaluator_ib1.evaluate(ib1_val_labels, pred_labels_ib1)

##########################
# TASK 3 — ground_truth.xlsx (9-class emotions)
##########################

gt_df = data_loader.load_dataset("ground_truth.csv")
gt_mapping = {
    'admiration': 0, 'amusement': 1, 'anger': 2, 'approval': 3, 'caring': 4,
    'curiosity': 5, 'disappointment': 6, 'excitement': 7, 'gratitude': 8
}
gt_df = data_loader.map_labels(gt_df, 'final_emotion', 'label_num', gt_mapping)

gt_train_texts, gt_val_texts, gt_train_labels, gt_val_labels = train_test_split(
    gt_df['text'], gt_df['label_num'], test_size=0.2, random_state=42
)

model_name_gt = "xlm-roberta-base"
tokenizer_gt = AutoTokenizer.from_pretrained(model_name_gt)
train_encodings_gt = tokenizer_gt(list(gt_train_texts), truncation=True, padding=True)
val_encodings_gt = tokenizer_gt(list(gt_val_texts), truncation=True, padding=True)

train_dataset_gt = CustomDataset(train_encodings_gt, list(gt_train_labels))
val_dataset_gt = CustomDataset(val_encodings_gt, list(gt_val_labels))

model_gt = AutoModelForSequenceClassification.from_pretrained(model_name_gt, num_labels=9)

trainer_gt = ModelTrainer(model_gt, train_dataset_gt, val_dataset_gt, results_dir / "ground_truth", epochs=3, batch_size=8)
trainer_gt.train()
trainer_gt.save_model(results_dir / "ground_truth_model")

model_gt.eval()
val_preds_gt = trainer_gt.trainer.predict(val_dataset_gt)
pred_labels_gt = torch.argmax(torch.tensor(val_preds_gt.predictions), axis=1).numpy()

evaluator_gt = ModelEvaluator(list(gt_mapping.keys()))
evaluator_gt.evaluate(gt_val_labels, pred_labels_gt)

print(" Full pipeline completed successfully!")
