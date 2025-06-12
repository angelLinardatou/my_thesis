import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import joblib
import warnings

from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier

warnings.filterwarnings("ignore")

# ----------------------------
# SETUP PATHS
# ----------------------------

base_dir = Path(__file__).parent
figure_dir = base_dir / "figures"
results_dir = base_dir / "results"
saved_models_dir = base_dir / "saved_models"

figure_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)
saved_models_dir.mkdir(exist_ok=True)

# ----------------------------
# LOAD DATA (USE eng.xlsx)
# ----------------------------

data_path = base_dir / "eng.xlsx"
df = pd.read_excel(data_path)

emotion_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']
X = df['text']
Y = df[emotion_cols]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Binarize multilabel outputs
mlb = MultiLabelBinarizer()
Y_train_bin = mlb.fit_transform(Y_train.values)
Y_test_bin = mlb.transform(Y_test.values)

# ----------------------------
# TRANSFORMER EMBEDDING FUNCTION
# ----------------------------

def extract_embeddings(model_name, texts):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embeddings = []
    for text in tqdm(texts, desc=f"Extracting embeddings for {model_name}"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(cls_embedding)

    return np.array(embeddings)

# ----------------------------
# DEFINE TRANSFORMERS
# ----------------------------

transformer_models = {
    "BERT": "bert-base-uncased",
    "RoBERTa": "roberta-base",
    "XLM-RoBERTa": "xlm-roberta-large",
    "ModernBERT": "nlpaueb/bert-base-greek-uncased-v1"
}

# ----------------------------
# LOOP THROUGH TRANSFORMERS
# ----------------------------

for name, model_path in transformer_models.items():
    print(f"\nProcessing {name} model...")

    # Extract embeddings
    X_train_emb = extract_embeddings(model_path, X_train)
    X_test_emb = extract_embeddings(model_path, X_test)

    # Save embeddings
    np.save(results_dir / f"{name}_train_embeddings.npy", X_train_emb)
    np.save(results_dir / f"{name}_test_embeddings.npy", X_test_emb)

    # ----------------------------
    # Train ML models on embeddings
    # ----------------------------

    models = {
        "LogisticRegression": MultiOutputClassifier(LogisticRegression(max_iter=1000)),
        "RandomForest": MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)),
        "SVM": MultiOutputClassifier(SVC(probability=True))
    }

    for model_name, clf in models.items():
        print(f"Training {model_name} on {name} embeddings...")
        clf.fit(X_train_emb, Y_train_bin)
        preds = clf.predict(X_test_emb)

        report = classification_report(Y_test_bin, preds, target_names=emotion_cols, output_dict=True)
        pd.DataFrame(report).to_csv(results_dir / f"{name}_{model_name}_report.csv")

        # Save trained model
        model_path_save = saved_models_dir / f"{name}_{model_name}.joblib"
        joblib.dump(clf, model_path_save)

print("\nAll transformers fully processed and saved.")
