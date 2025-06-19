from pathlib import Path
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from src.data_loader import DataLoader
from src.text_cleaner import TextCleaner
from src.transformer_embedding_extractor import EmbeddingExtractor
from src.transformer_trainer import TransformerTrainer
from src.evaluation import Evaluator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Paths
base_dir = Path(__file__).parent
data_dir = base_dir
results_dir = base_dir / "results_transformers"
results_dir.mkdir(exist_ok=True, parents=True)
figures_dir = base_dir / "figures"
figures_dir.mkdir(exist_ok=True)

# Load data
loader = DataLoader(data_dir)
df = loader.load_dataset("eng.csv")

# Clean text
cleaner = TextCleaner()
df['clean_text'] = df['text'].apply(cleaner.clean_text)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(df['clean_text'], df[['anger', 'fear', 'joy', 'sadness', 'surprise']], test_size=0.2, random_state=42)

# Binarize multilabel output
mlb = MultiLabelBinarizer()
Y_train_bin = mlb.fit_transform(Y_train.values)
Y_test_bin = mlb.transform(Y_test.values)

# Extract Transformer embeddings
transformer_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # You can adjust
extractor = EmbeddingExtractor(transformer_model)
X_train_embeddings = extractor.extract_embeddings(X_train)
X_test_embeddings = extractor.extract_embeddings(X_test)

# Train Models
trainer = TransformerTrainer()

# Train on embeddings
trainer.train_logistic_regression(X_train_embeddings, Y_train_bin)
trainer.train_random_forest(X_train_embeddings, Y_train_bin)
trainer.train_svm(X_train_embeddings, Y_train_bin)

# Evaluate Models
evaluator = Evaluator(['anger', 'fear', 'joy', 'sadness', 'surprise'])
for model_name in trainer.models.keys():
    preds = trainer.predict(model_name, X_test_embeddings)
    evaluator.evaluate_and_save(Y_test_bin, preds, results_dir / f"{model_name}_transformer_report.csv")

print("Full transformer pipeline completed successfully!")
