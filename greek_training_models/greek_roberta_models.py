from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pathlib import Path
base_dir = Path(__file__).parent
figure_dir = base_dir / 'figures'
figure_dir.mkdir(exist_ok=True)
results_dir = base_dir / 'results'
results_dir.mkdir(exist_ok=True)

#!/usr/bin/env python
# coding: utf-8

# In[1]:


#EDA for the gr.csv file 


# Load csv file 
file_path = "C:\\Users\\Aggeliki\\Desktop\\Thesis\\gr.csv"

#Read first 5 lines 
df = pd.read_csv(file_path)
print(df.head())


# In[7]:



# Load a CSV file and create a bar plot showing sentiment distribution

# Load the CSV file
file_path = "C:\\Users\\Aggeliki\\Desktop\\Thesis\\gr.csv"
df = pd.read_csv(file_path)

# Map the gold_label values to sentiment categories
df["gold_label_mapped"] = df["gold_label"].map({-1: "Negative", 0: "Neutral", 1: "Positive"})

# Count the occurrences of each sentiment category
sentiment_counts = df["gold_label_mapped"].value_counts()

# Define custom colors
colors = {"Negative": "black", "Neutral": "gray", "Positive": "blue"}

# Total number of data points
total_data = len(df)
num_negative = sentiment_counts.get("Negative", 0)
num_neutral = sentiment_counts.get("Neutral", 0)
num_positive = sentiment_counts.get("Positive", 0)

# Print sentiment statistics
print(f"Total data points: {total_data}")
print(f"Positive: {num_positive}")
print(f"Neutral: {num_neutral}")
print(f"Negative: {num_negative}")

# Create a bar plot with specified colors and transparency
plt.figure(figsize=(8, 6))
plt.bar(sentiment_counts.index, sentiment_counts.values, color=[colors[label] for label in sentiment_counts.index], alpha=0.7)
plt.xlabel("Sentiment Category")
plt.ylabel("Count")
plt.title("Sentiment Distribution in Data")
plt.savefig(figure_dir / 'plot_04.png')
plt.close()


# In[13]:


#Total NaN values 


# Define the file path
file_path = "C:\\Users\\Aggeliki\\Desktop\\Thesis\\gr.csv"

# Load the CSV file
df = pd.read_csv(file_path)

# Count NaN values in the 'gold_label' column
nan_gold_label = df['gold_label'].isna().sum()

# Print the result
print(f"NaN values in 'gold_label' column: {nan_gold_label}")


# In[9]:


#gr.csv data with fine tuning 


# Load the dataset
file_path_gr = "C:\\Users\\Aggeliki\\Desktop\\Thesis\\gr.csv"

data_gr = pd.read_csv(file_path_gr)

# Select text and labels
X_gr = data_gr["full_text"]
Y_gr = data_gr["gold_label"].map({-1: 0, 0: 1, 1: 2})  # Convert labels to {0,1,2}

# Split dataset
X_train_text_gr, X_test_text_gr, Y_train_gr, Y_test_gr = train_test_split(
    X_gr, Y_gr, test_size=0.2, random_state=42
)

# Load XLM-RoBERTa tokenizer and model
model_name = "xlm-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define PyTorch Dataset
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
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Create PyTorch datasets
train_dataset_gr = SentimentDataset(X_train_text_gr, Y_train_gr, tokenizer)
test_dataset_gr = SentimentDataset(X_test_text_gr, Y_test_gr, tokenizer)

# Create DataLoaders
train_loader_gr = DataLoader(train_dataset_gr, batch_size=16, shuffle=True)
test_loader_gr = DataLoader(test_dataset_gr, batch_size=16)

# Training Setup
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

epochs = 7
for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader_gr, desc=f"Epoch {epoch+1}")
    
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)

        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

# Evaluation
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_loader_gr:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        predictions.extend(preds)
        true_labels.extend(labels)

# Compute Metrics
accuracy = accuracy_score(true_labels, predictions)
precision, recall, f1_macro, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')

# Print classification report
print(classification_report(true_labels, predictions, target_names=["Negative", "Neutral", "Positive"]))
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro): {recall:.4f}")
print(f"F1 Score (Macro): {f1_macro:.4f}")


# In[7]:


#gr.csv without fine tuning 


# Load the dataset
file_path = "C:/Users/Aggeliki/Desktop/Thesis/gr.csv"  # Update path if needed
data = pd.read_csv(file_path)

# Extract text and labels
texts = data["full_text"].tolist()  # Use the correct text column
true_labels = data["gold_label"].tolist()  # Use the correct label column

# Convert labels to numerical format
label_mapping = {-1: 0, 0: 1, 1: 2}  # Adjust this if needed
true_labels = [label_mapping[label] for label in true_labels]

# Load pre-trained XLM-RoBERTa tokenizer and model
model_name = "xlm-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Adjusted to 3 labels

# Use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set model to evaluation mode

# Tokenization and inference with progress bar
predictions = []
batch_size = 16

print("\nStarting inference on gr.csv...")

for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
    batch_texts = texts[i:i+batch_size]
    
    # Convert texts to tokenized inputs
    inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    
    # Move input tensors to the selected device (CPU/GPU)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).cpu().numpy()  # Get the highest probability class
    predictions.extend(preds)

# Compute evaluation metrics with progress bar
print("\nCalculating evaluation metrics...")

with tqdm(total=4, desc="Computing Metrics") as pbar:
    accuracy = accuracy_score(true_labels, predictions)
    pbar.update(1)

    precision, recall, f1_macro, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')
    pbar.update(3)

# Print evaluation report
print("\nEvaluation Results:")
print(classification_report(true_labels, predictions, target_names=["Negative", "Neutral", "Positive"]))
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro): {recall:.4f}")
print(f"F1 Score (Macro): {f1_macro:.4f}")


# In[8]:


#EDA for the ib1_sentiment_probs.csv Omiros file 


# Define the file path
file_path_gr = "C:/Users/Aggeliki/Desktop/Thesis/ib1_sentiment_probs.csv"

df_gr = pd.read_csv(file_path_gr)
print(df_gr.head())


# In[10]:



# Define the file path
file_path_gr = "C:/Users/Aggeliki/Desktop/Thesis/ib1_sentiment_probs.csv"

# Load the CSV file
df_gr = pd.read_csv(file_path_gr)

# Determine the label based on the highest probability
df_gr["label"] = df_gr[["negative", "neutral", "positive", "narrator"]].idxmax(axis=1)

# Count occurrences of each label
label_counts = df_gr["label"].value_counts()

# Total number of data points
total_data = len(df_gr)
num_negative = label_counts.get("negative", 0)
num_neutral = label_counts.get("neutral", 0)
num_positive = label_counts.get("positive", 0)
num_narrator = label_counts.get("narrator", 0)

# Print sentiment statistics
print(f"Total data points: {total_data}")
print(f"Negative: {num_negative}")
print(f"Neutral: {num_neutral}")
print(f"Positive: {num_positive}")
print(f"Narrator: {num_narrator}")

# Define custom colors
colors = {"negative": "red", "neutral": "gray", "positive": "green", "narrator": "blue"}

# Create a bar plot with specified colors and transparency
plt.figure(figsize=(8, 6))
plt.bar(label_counts.index, label_counts.values, color=[colors[label] for label in label_counts.index], alpha=0.7)
plt.xlabel("Label Category")
plt.ylabel("Count")
plt.title("Label Distribution in Data")
plt.savefig(figure_dir / 'plot_05.png')
plt.close()


# In[11]:


#Total NaN values 


# Define the file path
file_path_gr = "C:/Users/Aggeliki/Desktop/Thesis/ib1_sentiment_probs.csv"

# Load the CSV file
df_gr = pd.read_csv(file_path_gr)

# Count total NaN values per column
nan_counts = df_gr.isna().sum()

# Count total NaN values in the entire dataset
total_nans = df_gr.isna().sum().sum()

# Print results
print("NaN values per column:")
print(nan_counts)
print(f"\nTotal NaN values in dataset: {total_nans}")


# In[6]:


#Omiros csv with fine tuning 


# Load the dataset
file_path_gr = "C:/Users/Aggeliki/Desktop/Thesis/ib1_sentiment_probs.csv"
data_gr = pd.read_csv(file_path_gr)

# Select text and labels
X_gr = data_gr["text"]  # Corrected column name

# Keep all categories including narrator
Y_gr = data_gr[["neutral", "positive", "negative", "narrator"]].idxmax(axis=1)

# Convert labels to numeric values
label_mapping = {"neutral": 0, "positive": 1, "negative": 2, "narrator": 3}
Y_gr = Y_gr.map(label_mapping)

# Split dataset
X_train_text_gr, X_test_text_gr, Y_train_gr, Y_test_gr = train_test_split(
    X_gr, Y_gr, test_size=0.2, random_state=42
)

# Load XLM-RoBERTa tokenizer and model
model_name = "xlm-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define PyTorch Dataset
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
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Create PyTorch datasets
train_dataset_gr = SentimentDataset(X_train_text_gr, Y_train_gr, tokenizer)
test_dataset_gr = SentimentDataset(X_test_text_gr, Y_test_gr, tokenizer)

# Create DataLoaders
train_loader_gr = DataLoader(train_dataset_gr, batch_size=16, shuffle=True)
test_loader_gr = DataLoader(test_dataset_gr, batch_size=16)

# Training Setup
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

epochs = 7
for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader_gr, desc=f"Epoch {epoch+1}")
    
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)

        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

# Evaluation
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_loader_gr:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        predictions.extend(preds)
        true_labels.extend(labels)

# Compute Metrics
accuracy = accuracy_score(true_labels, predictions)
precision, recall, f1_macro, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')

# Print classification report
print(classification_report(true_labels, predictions, target_names=["Neutral", "Positive", "Negative", "Narrator"]))
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro): {recall:.4f}")
print(f"F1 Score (Macro): {f1_macro:.4f}")


# In[5]:


#Omiros csv without fine tuning 


# Load the dataset
file_path = "C:/Users/Aggeliki/Desktop/Thesis/ib1_sentiment_probs.csv"
data = pd.read_csv(file_path)

# Extract text and corresponding labels
texts = data["text"].tolist()

# Convert labels to numerical values
label_mapping = {"neutral": 0, "positive": 1, "negative": 2, "narrator": 3}
true_labels = data[["neutral", "positive", "negative", "narrator"]].idxmax(axis=1).map(label_mapping).tolist()

# Load pre-trained XLM-RoBERTa tokenizer and model
model_name = "xlm-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set model to evaluation mode

# Tokenization and inference with progress bar
predictions = []
batch_size = 16

print("\nStarting inference...")

for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
    batch_texts = texts[i:i+batch_size]
    
    # Convert texts to tokenized inputs
    inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    
    # Move input tensors to the selected device (CPU/GPU)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).cpu().numpy()  # Get the highest probability class
    predictions.extend(preds)

# Compute evaluation metrics with progress bar
print("\nCalculating evaluation metrics...")

with tqdm(total=4, desc="Computing Metrics") as pbar:
    accuracy = accuracy_score(true_labels, predictions)
    pbar.update(1)

    precision, recall, f1_macro, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')
    pbar.update(3)

# Print evaluation report
print("\nEvaluation Results:")
print(classification_report(true_labels, predictions, target_names=["Neutral", "Positive", "Negative", "Narrator"]))
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro): {recall:.4f}")
print(f"F1 Score (Macro): {f1_macro:.4f}")


# In[15]:


#EDA for ground_truth.csv file 


# Load dataset
file_path = "C:\\Users\\Aggeliki\\Desktop\\Thesis\\Emotion_greek_data\\ground_truth.csv"
df = pd.read_csv(file_path)

print(df.head())


# In[27]:



file_path = "C:\\Users\\Aggeliki\\Desktop\\Thesis\\Emotion_greek_data\\ground_truth.csv"
df = pd.read_csv(file_path)


emotion_columns = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'none']

emotion_counts = df[emotion_columns].sum()

plt.figure(figsize=(10, 6))
plt.bar(emotion_counts.index, emotion_counts.values)
plt.xlabel("Emotion Labels")
plt.ylabel("Count of Texts")
plt.title("Count of Texts per Emotion Label")
plt.xticks(rotation=45)
plt.savefig(figure_dir / 'plot_06.png')
plt.close()


# In[18]:



# Load the dataset (replace the path if necessary)
file_path = "C:\\Users\\Aggeliki\\Desktop\\Thesis\\Emotion_greek_data\\ground_truth.csv"
df = pd.read_csv(file_path)

# Check for NaN values
nan_counts = df.isna().sum()

# Filter only columns with NaN values
nan_counts = nan_counts[nan_counts > 0]

# Display results
if nan_counts.empty:
    print("No NaN values found in the dataset.")
else:
    print("Columns with NaN values:")
    print(nan_counts)


# In[3]:



# Load the dataset
file_path = "C:\\Users\\Aggeliki\\Desktop\\Thesis\\Emotion_greek_data\\ground_truth.csv"
data = pd.read_csv(file_path)

# Select text and labels
X = data["text"].fillna(" ")  # Ensure no NaN values in text

# Define emotion labels including "none"
label_columns = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust", "none"]

# Ensure label columns are numeric
data[label_columns[:-1]] = data[label_columns[:-1]].apply(pd.to_numeric, errors="coerce")

# Fill NaN values with 0
data[label_columns[:-1]] = data[label_columns[:-1]].fillna(0)

# Identify the dominant emotion per row
Y = data[label_columns[:-1]].idxmax(axis=1)

# If all emotion values are zero, assign "none"
row_max_values = data[label_columns[:-1]].max(axis=1)
Y[row_max_values == 0] = "none"

# Convert labels to numeric values
label_mapping = {label: idx for idx, label in enumerate(label_columns)}
Y = Y.map(label_mapping)

# Check class distribution
print("Class Distribution before Oversampling:")
print(Y.value_counts())

# Split dataset
X_train_text, X_test_text, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# Apply Oversampling to balance the classes
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, Y_train_resampled = oversampler.fit_resample(X_train_text.to_frame(), Y_train)

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(Y_train_resampled), 
    y=Y_train_resampled.tolist()
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to("cuda" if torch.cuda.is_available() else "cpu")

# Load XLM-RoBERTa tokenizer and model
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_columns))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define PyTorch Dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Create PyTorch datasets
train_dataset = SentimentDataset(X_train_resampled["text"], Y_train_resampled, tokenizer)
test_dataset = SentimentDataset(X_test_text, Y_test, tokenizer)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Training Setup
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

epochs = 25
for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True, position=0)
    
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)

        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

# Evaluation
model.eval()
predictions = []
true_labels = []

test_loop = tqdm(test_loader, desc="Evaluating", leave=True, position=0)
with torch.no_grad():
    for batch in test_loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        predictions.extend(preds)
        true_labels.extend(labels)

# Compute Metrics
accuracy = accuracy_score(true_labels, predictions)
precision, recall, f1_macro, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')

# Dynamically get the unique labels from the test data
unique_labels = sorted(set(true_labels))  # Ensure correct ordering

# Adjust target_names to match only the present labels
adjusted_target_names = [label_columns[i] for i in unique_labels]

# Print classification report
print(classification_report(true_labels, predictions, labels=unique_labels, target_names=adjusted_target_names))
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro): {recall:.4f}")
print(f"F1 Score (Macro): {f1_macro:.4f}")


# In[2]:



# Load the dataset
file_path = "C:\\Users\\Aggeliki\\Desktop\\Thesis\\Emotion_greek_data\\ground_truth.csv"
data = pd.read_csv(file_path)

# Extract text and corresponding labels
texts = data["clean"].tolist()

# Define emotion labels including "none"
label_columns = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust", "none"]
true_labels = data[label_columns].idxmax(axis=1).tolist()

# Convert labels to numerical values
label_mapping = {label: idx for idx, label in enumerate(label_columns)}
numeric_labels = [label_mapping[label] for label in true_labels]

# Load pre-trained XLM-RoBERTa tokenizer and model
model_name = "xlm-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_columns))

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set model to evaluation mode

# Tokenization and inference with progress bar
predictions = []
batch_size = 16

print("\nStarting inference...")

for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
    batch_texts = texts[i:i+batch_size]
    
    # Convert texts to tokenized inputs
    inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    
    # Move input tensors to the selected device (CPU/GPU)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).cpu().numpy()  # Get the highest probability class
    predictions.extend(preds)

# Compute evaluation metrics with progress bar
print("\nCalculating evaluation metrics...")

with tqdm(total=4, desc="Computing Metrics") as pbar:
    accuracy = accuracy_score(numeric_labels, predictions)
    pbar.update(1)

    precision, recall, f1_macro, _ = precision_recall_fscore_support(numeric_labels, predictions, average='macro')
    pbar.update(3)

# Print evaluation report
print("\nEvaluation Results:")
print(classification_report(numeric_labels, predictions, target_names=label_columns))
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro): {recall:.4f}")
print(f"F1 Score (Macro): {f1_macro:.4f}")


# In[ ]:



