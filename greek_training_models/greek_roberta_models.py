import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from pathlib import Path

# =========================================
# Setup directories
# =========================================

# Define base directory as the folder containing this script
base_dir = Path(__file__).parent
annotation_dir = base_dir / 'annotations'
figure_dir = base_dir / 'figures'
results_dir = base_dir / 'results'

# Create directories if they don't exist
figure_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

# =========================================
# Dataset 1: gr.csv - Sentiment (3-class)
# =========================================

# Load dataset
file_path = annotation_dir / 'gr.csv'
df = pd.read_csv(file_path)

# Map sentiment labels
mapping_gr = {-1: "Negative", 0: "Neutral", 1: "Positive"}
df["gold_label_mapped"] = df["gold_label"].map(mapping_gr)

# Plot sentiment distribution
sentiment_counts = df["gold_label_mapped"].value_counts()
plt.figure(figsize=(8, 6))
plt.bar(sentiment_counts.index, sentiment_counts.values, color=["black", "gray", "blue"], alpha=0.7)
plt.xlabel("Sentiment Category")
plt.ylabel("Count")
plt.title("Sentiment Distribution in gr.csv")
plt.savefig(figure_dir / 'plot_01_sentiment_distribution.png')
plt.close()

# Prepare train-test split
X_gr = df["full_text"]
Y_gr = df["gold_label"].map({-1: 0, 0: 1, 1: 2})
X_train_gr, X_test_gr, Y_train_gr, Y_test_gr = train_test_split(X_gr, Y_gr, test_size=0.2, random_state=42)

# Load tokenizer and model
model_name_gr = "xlm-roberta-large"
tokenizer_gr = AutoTokenizer.from_pretrained(model_name_gr)
model_gr = AutoModelForSequenceClassification.from_pretrained(model_name_gr, num_labels=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_gr.to(device)

# Define custom dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        return {"input_ids": inputs["input_ids"].squeeze(0), "attention_mask": inputs["attention_mask"].squeeze(0), "labels": torch.tensor(label, dtype=torch.long)}

# Prepare dataloaders
train_loader_gr = DataLoader(SentimentDataset(X_train_gr, Y_train_gr, tokenizer_gr), batch_size=16, shuffle=True)
test_loader_gr = DataLoader(SentimentDataset(X_test_gr, Y_test_gr, tokenizer_gr), batch_size=16)

# Train model
optimizer = AdamW(model_gr.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(7):
    model_gr.train()
    loop = tqdm(train_loader_gr, desc=f"gr.csv Epoch {epoch+1}")
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model_gr(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

# Evaluate model
model_gr.eval()
predictions, true_labels = [], []
with torch.no_grad():
    for batch in test_loader_gr:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()
        outputs = model_gr(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels)

# Save evaluation results
report = classification_report(true_labels, predictions, output_dict=True)
pd.DataFrame(report).to_csv(results_dir / "gr_classification_report.csv")

# =========================================
# Dataset 2: ib1_sentiment_probs.csv - 4-class sentiment
# =========================================

file_path_ib1 = annotation_dir / 'ib1_sentiment_probs.csv'
df_ib1 = pd.read_csv(file_path_ib1)
df_ib1["label"] = df_ib1[["negative", "neutral", "positive", "narrator"]].idxmax(axis=1)

# Plot label distribution
label_counts = df_ib1["label"].value_counts()
plt.figure(figsize=(8, 6))
plt.bar(label_counts.index, label_counts.values, color=["red", "gray", "green", "blue"], alpha=0.7)
plt.xlabel("Label Category")
plt.ylabel("Count")
plt.title("Label Distribution in ib1_sentiment_probs.csv")
plt.savefig(figure_dir / 'plot_02_label_distribution.png')
plt.close()

# Prepare train-test split
X_ib1 = df_ib1["text"]
Y_ib1 = df_ib1["label"].map({"neutral": 0, "positive": 1, "negative": 2, "narrator": 3})
X_train_ib1, X_test_ib1, Y_train_ib1, Y_test_ib1 = train_test_split(X_ib1, Y_ib1, test_size=0.2, random_state=42)

model_ib1 = AutoModelForSequenceClassification.from_pretrained(model_name_gr, num_labels=4)
model_ib1.to(device)

train_loader_ib1 = DataLoader(SentimentDataset(X_train_ib1, Y_train_ib1, tokenizer_gr), batch_size=16, shuffle=True)
test_loader_ib1 = DataLoader(SentimentDataset(X_test_ib1, Y_test_ib1, tokenizer_gr), batch_size=16)

optimizer = AdamW(model_ib1.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(7):
    model_ib1.train()
    loop = tqdm(train_loader_ib1, desc=f"ib1.csv Epoch {epoch+1}")
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model_ib1(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

model_ib1.eval()
predictions, true_labels = [], []
with torch.no_grad():
    for batch in test_loader_ib1:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()
        outputs = model_ib1(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels)

report = classification_report(true_labels, predictions, output_dict=True)
pd.DataFrame(report).to_csv(results_dir / "ib1_classification_report.csv")

# =========================================
# Dataset 3: ground_truth.csv - Emotion (9-class)
# =========================================

file_path_gt = annotation_dir / 'ground_truth.csv'
df_gt = pd.read_csv(file_path_gt)
emotion_columns = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'none']
df_gt[emotion_columns] = df_gt[emotion_columns].fillna(0)
df_gt['max_label'] = df_gt[emotion_columns].idxmax(axis=1)
label_mapping = {label: idx for idx, label in enumerate(emotion_columns)}
df_gt['label'] = df_gt['max_label'].map(label_mapping)

# Plot emotion distribution
emotion_counts = df_gt[emotion_columns].sum()
plt.figure(figsize=(10, 6))
plt.bar(emotion_counts.index, emotion_counts.values)
plt.xlabel("Emotion Labels")
plt.ylabel("Count")
plt.title("Emotion Distribution in ground_truth.csv")
plt.xticks(rotation=45)
plt.savefig(figure_dir / 'plot_03_emotion_distribution.png')
plt.close()

# Train-test split with oversampling
X_gt = df_gt["text"]
Y_gt = df_gt['label']
X_train_gt, X_test_gt, Y_train_gt, Y_test_gt = train_test_split(X_gt, Y_gt, test_size=0.2, random_state=42)
oversampler = RandomOverSampler()
X_train_gt_resampled, Y_train_gt_resampled = oversampler.fit_resample(X_train_gt.to_frame(), Y_train_gt)

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train_gt_resampled), y=Y_train_gt_resampled)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

model_name_gt = "xlm-roberta-base"
tokenizer_gt = AutoTokenizer.from_pretrained(model_name_gt)
model_gt = AutoModelForSequenceClassification.from_pretrained(model_name_gt, num_labels=len(emotion_columns))
model_gt.to(device)

train_loader_gt = DataLoader(SentimentDataset(X_train_gt_resampled["text"], Y_train_gt_resampled, tokenizer_gt), batch_size=16, shuffle=True)
test_loader_gt = DataLoader(SentimentDataset(X_test_gt, Y_test_gt, tokenizer_gt), batch_size=16)

optimizer = AdamW(model_gt.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
for epoch in range(7):
    model_gt.train()
    loop = tqdm(train_loader_gt, desc=f"ground_truth.csv Epoch {epoch+1}")
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model_gt(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

model_gt.eval()
predictions, true_labels = [], []
with torch.no_grad():
    for batch in test_loader_gt:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()
        outputs = model_gt(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels)

report = classification_report(true_labels, predictions, output_dict=True)
pd.DataFrame(report).to_csv(results_dir / "ground_truth_classification_report.csv")