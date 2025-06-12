import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string
import nltk
import optuna

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.multioutput import MultiOutputClassifier
from gensim.models import Word2Vec
from sklearn.preprocessing import MultiLabelBinarizer

nltk.download('stopwords')
from nltk.corpus import stopwords

# ----------------------------
# SETUP PATHS
# ----------------------------

base_dir = Path(__file__).parent
figure_dir = base_dir / "figures"
results_dir = base_dir / "results"

figure_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

# ----------------------------
# LOAD DATA (USE eng.xlsx)
# ----------------------------

data_path = base_dir / "eng.xlsx"
df = pd.read_excel(data_path)

emotion_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']
X = df['text']
Y = df[emotion_cols]

# ----------------------------
# TEXT CLEANING FUNCTION
# ----------------------------

stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

X_clean = X.apply(clean_text)

# ----------------------------
# TRAIN TEST SPLIT
# ----------------------------

X_train, X_test, Y_train, Y_test = train_test_split(X_clean, Y, test_size=0.2, random_state=42)

# ----------------------------
# EDA PLOT CLASS DISTRIBUTION
# ----------------------------

class_counts = Y.sum()
plt.figure(figsize=(8,5))
class_counts.plot(kind='bar')
plt.title("Class Distribution")
plt.savefig(figure_dir / "class_distribution.png")
plt.close()

# ----------------------------
# TF-IDF FEATURE EXTRACTION
# ----------------------------

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ----------------------------
# MULTI-LABEL BINARIZER
# ----------------------------

mlb = MultiLabelBinarizer()
Y_train_bin = mlb.fit_transform(Y_train.values)
Y_test_bin = mlb.transform(Y_test.values)

# ----------------------------
# LOGISTIC REGRESSION + OPTUNA
# ----------------------------

def objective(trial):
    C = trial.suggest_loguniform("C", 1e-3, 1e2)
    model = MultiOutputClassifier(LogisticRegression(C=C, max_iter=1000))
    model.fit(X_train_tfidf, Y_train_bin)
    preds = model.predict(X_test_tfidf)
    return f1_score(Y_test_bin, preds, average="micro")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

best_C = study.best_params['C']
log_reg = MultiOutputClassifier(LogisticRegression(C=best_C, max_iter=1000))
log_reg.fit(X_train_tfidf, Y_train_bin)
preds_log = log_reg.predict(X_test_tfidf)

report_log = classification_report(Y_test_bin, preds_log, target_names=emotion_cols, output_dict=True)
pd.DataFrame(report_log).to_csv(results_dir / "logistic_regression_report.csv")

# ----------------------------
# SUPPORT VECTOR MACHINE + OPTUNA
# ----------------------------

def objective_svm(trial):
    C = trial.suggest_loguniform("C", 1e-3, 1e2)
    kernel = trial.suggest_categorical("kernel", ['linear', 'rbf'])
    model = MultiOutputClassifier(SVC(C=C, kernel=kernel, probability=True))
    model.fit(X_train_tfidf, Y_train_bin)
    preds = model.predict(X_test_tfidf)
    return f1_score(Y_test_bin, preds, average="micro")

study_svm = optuna.create_study(direction="maximize")
study_svm.optimize(objective_svm, n_trials=30)

best_C_svm = study_svm.best_params['C']
best_kernel = study_svm.best_params['kernel']
svm = MultiOutputClassifier(SVC(C=best_C_svm, kernel=best_kernel, probability=True))
svm.fit(X_train_tfidf, Y_train_bin)
preds_svm = svm.predict(X_test_tfidf)

report_svm = classification_report(Y_test_bin, preds_svm, target_names=emotion_cols, output_dict=True)
pd.DataFrame(report_svm).to_csv(results_dir / "svm_report.csv")

# ----------------------------
# RANDOM FOREST
# ----------------------------

rf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
rf.fit(X_train_tfidf, Y_train_bin)
preds_rf = rf.predict(X_test_tfidf)

report_rf = classification_report(Y_test_bin, preds_rf, target_names=emotion_cols, output_dict=True)
pd.DataFrame(report_rf).to_csv(results_dir / "random_forest_report.csv")

# ----------------------------
# K-NEAREST NEIGHBORS
# ----------------------------

knn = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5))
knn.fit(X_train_tfidf, Y_train_bin)
preds_knn = knn.predict(X_test_tfidf)

report_knn = classification_report(Y_test_bin, preds_knn, target_names=emotion_cols, output_dict=True)
pd.DataFrame(report_knn).to_csv(results_dir / "knn_report.csv")

# ----------------------------
# WORD2VEC EMBEDDINGS
# ----------------------------

X_tokens = [text.split() for text in X_train]
w2v_model = Word2Vec(sentences=X_tokens, vector_size=100, window=5, min_count=1, workers=4)

# Sentence embeddings using mean pooling
def embed(text):
    tokens = text.split()
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

X_train_w2v = np.vstack([embed(text) for text in X_train])
X_test_w2v = np.vstack([embed(text) for text in X_test])

# Train model on Word2Vec embeddings
log_reg_w2v = MultiOutputClassifier(LogisticRegression(max_iter=1000))
log_reg_w2v.fit(X_train_w2v, Y_train_bin)
preds_w2v = log_reg_w2v.predict(X_test_w2v)

report_w2v = classification_report(Y_test_bin, preds_w2v, target_names=emotion_cols, output_dict=True)
pd.DataFrame(report_w2v).to_csv(results_dir / "word2vec_logreg_report.csv")

print("\nAll supervised models completed successfully. Reports & figures saved.")
