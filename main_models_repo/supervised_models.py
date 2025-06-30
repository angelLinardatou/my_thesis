from pathlib import Path
import pandas as pd
import nltk
from src.data_loader import load_dataset
from src.text_cleaner import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from src.features_word2vec import Word2VecFeatures
from src.supervised_trainer import SupervisedTrainer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

nltk.download('stopwords')

# Paths
base_dir = Path(__file__).parent
data_dir = base_dir
results_dir = base_dir / "results"
results_dir.mkdir(exist_ok=True, parents=True)
figures_dir = base_dir / "figures"
figures_dir.mkdir(exist_ok=True)

# Load data
df = load_dataset(data_dir, "eng.csv")

# Clean text
df['clean_text'] = df['text'].apply(clean_text)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(df['clean_text'], df[['anger', 'fear', 'joy', 'sadness', 'surprise']], test_size=0.2, random_state=42)

# Binarize multilabel output
mlb = MultiLabelBinarizer()
Y_train_bin = mlb.fit_transform(Y_train.values)
Y_test_bin = mlb.transform(Y_test.values)

# TF-IDF Features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Word2Vec Features
word2vec = Word2VecFeatures(vector_size=100)
tokenized_texts = [text.split() for text in X_train]
word2vec.train_word2vec(tokenized_texts)
X_train_w2v = word2vec.transform_dataset(X_train)
X_test_w2v = word2vec.transform_dataset(X_test)

# Train Models
trainer = SupervisedTrainer()

# Train on TF-IDF
trainer.train_logistic_regression(X_train_tfidf, Y_train_bin)
trainer.train_random_forest(X_train_tfidf, Y_train_bin)
trainer.train_svm(X_train_tfidf, Y_train_bin)
trainer.train_knn(X_train_tfidf, Y_train_bin)

label_names = ['anger', 'fear', 'joy', 'sadness', 'surprise']

# Evaluate TF-IDF Models
for model_name in trainer.models.keys():
    preds = trainer.predict(model_name, X_test_tfidf)
    report = classification_report(Y_test_bin, preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(results_dir / f"{model_name}_tfidf_report.csv")
    print(report_df)

# Train on Word2Vec (only example with logistic regression here)
trainer.train_logistic_regression(X_train_w2v, Y_train_bin)
preds_w2v = trainer.predict("LogisticRegression", X_test_w2v)
report = classification_report(Y_test_bin, preds_w2v, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(results_dir / "logistic_word2vec_report.csv")
print(report_df)

print(" Full supervised ML pipeline completed successfully!")
