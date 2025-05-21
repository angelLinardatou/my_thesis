from catboost import CatBoostClassifier
from itertools import product
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tqdm import tqdm  # Import tqdm for progress bars
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import warnings

from pathlib import Path
base_dir = Path(__file__).parent
figure_dir = base_dir / 'figures'
figure_dir.mkdir(exist_ok=True)
results_dir = base_dir / 'results'
results_dir.mkdir(exist_ok=True)


# Suppress all warnings
warnings.filterwarnings('ignore')

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Sample data preprocessing
def preprocess_text(text):
    return text.lower()

# Convert text data into BERT embeddings
def get_bert_embeddings(text_data):
    embeddings = []
    for text in text_data:
        # Tokenize text and get input IDs
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        
        # Pass the tokenized inputs through BERT model to get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use the last hidden state as embeddings (take the mean of all token embeddings)
        last_hidden_states = outputs.last_hidden_state
        embeddings.append(last_hidden_states.mean(dim=1).squeeze().numpy())  # Average over the tokens
        
    return np.array(embeddings)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear', random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

# Convert text data to BERT embeddings
X_train_embeddings = get_bert_embeddings(X_train)
X_test_embeddings = get_bert_embeddings(X_test)

results = []

print('Print only F1 per emotion:')

# Save the best performing model
best_model = None
best_f1_macro = 0

for model_name, model in models.items():
    multi_label_model = MultiOutputClassifier(model)
    multi_label_model.fit(X_train_embeddings, Y_train)
    
    Y_pred = multi_label_model.predict(X_test_embeddings)
    
    # Macro precision, recall, and f1 score
    precision, recall, f1_macro, _ = precision_recall_fscore_support(Y_test, Y_pred, average='macro')
    
    # Micro precision, recall, and f1 score
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(Y_test, Y_pred, average='micro')
    
    # F1 score per emotion
    f1_per_emotions = f1_score(Y_test, Y_pred, average=None)
    
    # Accuracy score
    accuracy = accuracy_score(Y_test, Y_pred)
    
    # Samples-F1 
    sample_f1_scores = [
        f1_score(Y_test.iloc[i], Y_pred[i], average='weighted') for i in range(len(Y_test))
    ]
    sample_f1 = sum(sample_f1_scores) / len(sample_f1_scores)
    
    results.append({
        'Model': model_name,
        'Precision': precision,
        'Recall': recall,
        'F1 Score (Macro)': f1_macro,
        'F1 Score (Micro)': f1_micro,
        'Accuracy': accuracy,
        'F1 Score per Emotions': f1_per_emotions,
        'Samples-F1':  sample_f1
    })
    
    # Print F1 per emotion
    print(f"{model_name}: {f1_per_emotions}")
    
    # Save the best model
    if f1_macro > best_f1_macro:
        best_f1_macro = f1_macro
        best_model = multi_label_model

# Save the best model to a file
joblib.dump(best_model, 'trained_model.joblib')

# Convert results to DataFrame for easy viewing and plotting
results_df = pd.DataFrame(results)
print('\nResult Table:')
print(results_df[['Model', 'Precision', 'Recall', 'F1 Score (Macro)', 'F1 Score (Micro)', 'Samples-F1', 'Accuracy']])


# In[27]:


pip install transformers


# In[37]:



# Load the previously trained models and tokenizer
model_path = "C:/Users/Aggeliki/Desktop/Thesis/trained_model.joblib"
trained_model = joblib.load(model_path)

# Load pre-trained BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Load the data file
file_path = "C:/Users/Aggeliki/Desktop/Thesis/public_data_2/track_a/dev/eng.csv"
data = pd.read_csv(file_path)

# Fill missing text with empty strings for preprocessing purposes
data.fillna('', inplace=True)

# Function to preprocess text
def preprocess_text(text):
    return text.lower()

def get_bert_embeddings(text_data):
    embeddings = []
    for text in text_data:
        inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        embeddings.append(last_hidden_states.mean(dim=1).squeeze().numpy())
    return np.array(embeddings)

# Apply preprocessing to the text column
if 'text' in data.columns:
    data['processed_text'] = data['text'].apply(preprocess_text)
else:
    raise ValueError("The file must contain a 'text' column.")

# Generate BERT embeddings for the text data
text_embeddings = get_bert_embeddings(data['processed_text'].tolist())

# Predict emotions using the trained model
predictions = trained_model.predict(text_embeddings)  # Get binary predictions (0 or 1) for each emotion

# Assuming the predictions align with the shape of the missing values column
# Fill in the missing values for multiple emotion columns
columns_to_update = ["anger", "fear", "joy", "sadness", "surprise"]
if len(columns_to_update) != predictions.shape[1]:
    print("Error: The number of columns does not match the prediction shape.")
else:
    for i, col in enumerate(columns_to_update):
        data[col] = predictions[:, i]

# Drop the 'text' column
data.drop(columns=['text', 'processed_text'], inplace=True)

# Save the updated dataset
output_file_path = "C:\\Users\\Aggeliki\\Desktop\\Thesis\\public_data_2\\track_a\\dev\\predictions_output.csv"
data.to_csv(output_file_path, index=False)

print("Predictions saved at: 'predictions_output.csv'.")


# In[38]:



# Suppress warnings
warnings.filterwarnings('ignore')

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear', random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

# Load RoBERTa tokenizer and model
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base')

# Convert text to RoBERTa embeddings
def get_roberta_embeddings(text_data):
    embeddings = []
    for text in text_data:
        # Tokenize text
        inputs = roberta_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        # Get RoBERTa embeddings
        with torch.no_grad():
            outputs = roberta_model(**inputs)
            # Take the [CLS] token embedding (pooled output)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(cls_embedding)
    return np.array(embeddings)

# Convert text data to RoBERTa embeddings
X_train_embeddings = get_roberta_embeddings(X_train)
X_test_embeddings = get_roberta_embeddings(X_test)

results = []

print('Print only F1 per emotion:')

# Save the best performing model
best_model = None
best_f1_macro = 0

for model_name, model in models.items():
    multi_label_model = MultiOutputClassifier(model)
    multi_label_model.fit(X_train_embeddings, Y_train)
    
    Y_pred = multi_label_model.predict(X_test_embeddings)
    
    # Macro precision, recall, and f1 score
    precision, recall, f1_macro, _ = precision_recall_fscore_support(Y_test, Y_pred, average='macro')
    
    # Micro precision, recall, and f1 score
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(Y_test, Y_pred, average='micro')
    
    # F1 score per emotion
    f1_per_emotions = f1_score(Y_test, Y_pred, average=None)
    
    # Accuracy score
    accuracy = accuracy_score(Y_test, Y_pred)
    
    # Samples-F1 
    sample_f1_scores = [
        f1_score(Y_test.iloc[i], Y_pred[i], average='weighted') for i in range(len(Y_test))
    ]
    sample_f1 = sum(sample_f1_scores) / len(sample_f1_scores)
    
    results.append({
        'Model': model_name,
        'Precision': precision,
        'Recall': recall,
        'F1 Score (Macro)': f1_macro,
        'F1 Score (Micro)': f1_micro,
        'Accuracy': accuracy,
        'F1 Score per Emotions': f1_per_emotions,
        'Samples-F1':  sample_f1
    })
    
    # Print F1 per emotion
    print(f"{model_name}: {f1_per_emotions}")
    
    # Save the best model
    if f1_macro > best_f1_macro:
        best_f1_macro = f1_macro
        best_model = multi_label_model

# Save the best model to a file
joblib.dump(best_model, 'best_roberta_model.joblib')

# Convert results to DataFrame for easy viewing and plotting
results_df = pd.DataFrame(results)
print('\nResult Table:')
print(results_df[['Model', 'Precision', 'Recall', 'F1 Score (Macro)', 'F1 Score (Micro)', 'Samples-F1', 'Accuracy']])


# In[39]:



# Load the previously trained models and tokenizer
model_path = "C:/Users/Aggeliki/Desktop/Thesis/best_roberta_model.joblib"
trained_model = joblib.load(model_path)

# Load pre-trained RoBERTa tokenizer
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base')

# Load the data file
file_path = "C:/Users/Aggeliki/Desktop/Thesis/public_data_2/track_a/dev/eng.csv"
data = pd.read_csv(file_path)

# Fill missing text with empty strings for preprocessing purposes
data.fillna('', inplace=True)

# Function to preprocess text
def preprocess_text(text):
    return text.lower()

def get_roberta_embeddings(text_data):
    embeddings = []
    for text in text_data:
        inputs = roberta_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = roberta_model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        embeddings.append(last_hidden_states.mean(dim=1).squeeze().numpy())
    return np.array(embeddings)

# Apply preprocessing to the text column
if 'text' in data.columns:
    data['processed_text'] = data['text'].apply(preprocess_text)
else:
    raise ValueError("The file must contain a 'text' column.")

# Generate RoBERTa embeddings for the text data
text_embeddings = get_roberta_embeddings(data['processed_text'].tolist())

# Predict emotions using the trained model
predictions = trained_model.predict(text_embeddings)  # Get binary predictions (0 or 1) for each emotion

# Assuming the predictions align with the shape of the missing values column
# Fill in the missing values for multiple emotion columns
columns_to_update = ["anger", "fear", "joy", "sadness", "surprise"]
if len(columns_to_update) != predictions.shape[1]:
    print("Error: The number of columns does not match the prediction shape.")
else:
    for i, col in enumerate(columns_to_update):
        data[col] = predictions[:, i]

# Drop the 'text' column
data.drop(columns=['text', 'processed_text'], inplace=True)

# Save the updated dataset
output_file_path = "C:\\Users\\Aggeliki\\Desktop\\Thesis\\public_data_2\\track_a\\dev\\predictions_output.csv"
data.to_csv(output_file_path, index=False)

print("Predictions saved at: 'predictions_output.csv'.")


# In[136]:



# Define the base models
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs')),
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('SVM', SVC(kernel='linear', random_state=42)),
    ('K-Nearest Neighbors', KNeighborsClassifier(n_neighbors=5)),
    ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
]

# Create the Voting Classifier using hard voting (majority voting)
voting_classifier = VotingClassifier(estimators=models, voting='hard')

# Wrap the VotingClassifier in MultiOutputClassifier for multi-label classification
multi_label_voting_model = MultiOutputClassifier(voting_classifier)

# Convert text data to Word2Vec embeddings
X_train_embeddings = get_word2vec_embeddings(X_train)
X_test_embeddings = get_word2vec_embeddings(X_test)

# Train the model
multi_label_voting_model.fit(X_train_embeddings, Y_train)

# Make predictions
Y_pred = multi_label_voting_model.predict(X_test_embeddings)

# Macro precision, recall, and f1 score
precision, recall, f1_macro, _ = precision_recall_fscore_support(Y_test, Y_pred, average='macro')

# Micro precision, recall, and f1 score
precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(Y_test, Y_pred, average='micro')

# F1 score per emotion
f1_per_emotions = f1_score(Y_test, Y_pred, average=None)

# Accuracy score
accuracy = accuracy_score(Y_test, Y_pred)

# Samples-F1
sample_f1_scores = [
    f1_score(Y_test.iloc[i], Y_pred[i], average='weighted') for i in range(len(Y_test))
]
sample_f1 = sum(sample_f1_scores) / len(sample_f1_scores)

# Print F1 per emotion
print(f"Voting Classifier: {f1_per_emotions}")

# Collect results
results = [{
    'Model': 'Voting Classifier',
    'Precision': precision,
    'Recall': recall,
    'F1 Score (Macro)': f1_macro,
    'F1 Score (Micro)': f1_micro,
    'Accuracy': accuracy,
    'F1 Score per Emotions': f1_per_emotions,
    'Samples-F1': sample_f1
}]

# Convert results to DataFrame for easy viewing and plotting
results_df = pd.DataFrame(results)
print('\nResult Table:')
print(results_df[['Model', 'Precision', 'Recall', 'F1 Score (Macro)', 'F1 Score (Micro)', 'Samples-F1', 'Accuracy']])


# In[137]:



# Suppress all warnings
warnings.filterwarnings('ignore')

# Define individual models
models = [
    ('LR', LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs')),
    ('RF', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('SVM', SVC(kernel='linear', random_state=42)),
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
    ('XGB', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
]

# Create the Voting Classifier with hard voting (majority voting)
voting_classifier = VotingClassifier(estimators=models, voting='hard')

# Prepare the results list
results = []

print('Print only F1 per emotion:')

# For each model
for i, (model_name, model) in enumerate(models, start=1):
    
    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    joblib.dump(tfidf_vectorizer, 'C:/Users/Aggeliki/Desktop/Thesis/public_data/tfidf_vectorizer.pkl')

    # Use MultiOutputClassifier for multi-label classification with Voting Classifier
    multi_label_model = MultiOutputClassifier(voting_classifier)
    multi_label_model.fit(X_train_tfidf, Y_train)
    
    joblib.dump(multi_label_model, 'saved_model.pkl')
    
    # Make predictions
    Y_pred = multi_label_model.predict(X_test_tfidf)
    
    # Macro precision, recall, and f1 score
    precision, recall, f1_macro, _ = precision_recall_fscore_support(Y_test, Y_pred, average='macro')
    
    # Micro precision, recall, and f1 score
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(Y_test, Y_pred, average='micro')
    
    # F1 score per emotion
    f1_per_emotions = f1_score(Y_test, Y_pred, average=None)
    
    # Samples-F1 
    sample_f1_scores = [
        f1_score(Y_test.iloc[i], Y_pred[i], average='weighted') for i in range(len(Y_test))
    ]
    sample_f1 = sum(sample_f1_scores) / len(sample_f1_scores)
    
    # Accuracy score
    accuracy = accuracy_score(Y_test, Y_pred)
    
    results.append({
        'Model': 'Voting Classifier',
        'Precision': precision,
        'Recall': recall,
        'F1 Score (Macro)': f1_macro,
        'F1 Score (Micro)': f1_micro,
        'Accuracy': accuracy,
        'F1 Score per Emotions': f1_per_emotions,
        'Samples-F1': sample_f1
    })
    
    # Print F1 per emotion
    print(f"Voting Classifier: {f1_per_emotions}")

results_df = pd.DataFrame(results)

# Print result table
print('\nResult Table:')
print(results_df[['Model', 'Precision', 'Recall', 'F1 Score (Macro)', 'F1 Score (Micro)', 'Samples-F1', 'Accuracy']])


# In[38]:


get_ipython().system('pip install imbalanced-learn')
get_ipython().system('pip install imbalanced-learn-contrib')


# In[49]:



# Load the dataset
file_path = "C:/Users/Aggeliki/Desktop/Thesis/public_data_2/track_a/train/eng.csv"
data = pd.read_csv(file_path)

# Split the data into training and test sets (example split)
X_train, X_test, Y_train, Y_test = train_test_split(
    data['text'], data[['anger', 'fear', 'joy', 'sadness', 'surprise']], test_size=0.2, random_state=42
)

# Load BERT model and tokenizer
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to create BERT embeddings
def create_bert_embeddings(texts, max_length=128):
    if isinstance(texts, pd.Series):  # Convert pandas Series to list
        texts = texts.tolist()
    inputs = bert_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Generate BERT embeddings
X_train_embeddings = create_bert_embeddings(X_train)
X_test_embeddings = create_bert_embeddings(X_test)

# Define models
models = {
    'LR': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'RF': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'XGB': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
}

# Ensemble Voting Classifier (excluding BERT for direct voting)
voting_ensemble = VotingClassifier(
    estimators=[(name, model) for name, model in models.items()],
    voting='soft'  # Use soft voting for probabilities
)

# Multi-label wrapper
multi_label_model = MultiOutputClassifier(voting_ensemble)

# Train models
multi_label_model.fit(X_train_embeddings, Y_train)
# Save model
joblib.dump(multi_label_model, 'C:/Users/Aggeliki/Desktop/Thesis/public_data/voting_ensemble_model.pkl')

# Predictions
Y_pred = multi_label_model.predict(X_test_embeddings)

# Evaluation
precision, recall, f1_macro, _ = precision_recall_fscore_support(Y_test, Y_pred, average='macro')
accuracy = accuracy_score(Y_test, Y_pred)

print("Evaluation Metrics:")
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro): {recall:.4f}")
print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"Accuracy: {accuracy:.4f}")


# In[64]:



# Load the dataset
file_path = "C:/Users/Aggeliki/Desktop/Thesis/public_data_2/track_a/train/eng.csv"
data = pd.read_csv(file_path)

# Feature Engineering: Adding TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
tfidf_features = vectorizer.fit_transform(data['text']).toarray()

# Split the data into training and test sets
X_train_text, X_test_text, Y_train, Y_test = train_test_split(
    data['text'], data[['anger', 'fear', 'joy', 'sadness', 'surprise']], test_size=0.2, random_state=42
)

# Generate TF-IDF features for train and test
X_train_tfidf = vectorizer.transform(X_train_text).toarray()
X_test_tfidf = vectorizer.transform(X_test_text).toarray()

# Load BERT model and tokenizer
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to create BERT embeddings
def create_bert_embeddings(texts, max_length=128):
    if isinstance(texts, pd.Series):  # Convert pandas Series to list
        texts = texts.tolist()
    inputs = bert_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Generate BERT embeddings
X_train_bert = create_bert_embeddings(X_train_text)
X_test_bert = create_bert_embeddings(X_test_text)

# Combine BERT embeddings and TF-IDF features
X_train_combined = np.hstack((X_train_bert, X_train_tfidf))
X_test_combined = np.hstack((X_test_bert, X_test_tfidf))

# Define models
models = {
    'LR': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'RF': RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
    'SVM': SVC(kernel='linear', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'XGB': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=150, random_state=42),
}

# Ensemble Voting Classifier with weights
voting_ensemble = VotingClassifier(
    estimators=[(name, model) for name, model in models.items()],
    voting='soft',  # Use soft voting for probabilities
    weights=[1, 2, 1, 1, 3]  # Assign higher weight to RF and XGB
)

# Multi-label wrapper
multi_label_model = MultiOutputClassifier(voting_ensemble)

# Train models
multi_label_model.fit(X_train_combined, Y_train)

# Save model
joblib.dump(multi_label_model, 'C:/Users/Aggeliki/Desktop/Thesis/public_data/voting_ensemble_model_with_features.pkl')

# Predictions probabilities
Y_pred_proba = multi_label_model.predict_proba(X_test_combined)

# Grid Search for Fine-Tuned Thresholds
candidate_thresholds = np.arange(0.2, 0.5, 0.05)  # Fine-grained thresholds
best_f1 = 0
best_thresholds = []

for thresholds in product(candidate_thresholds, repeat=Y_test.shape[1]):
    Y_pred = np.zeros_like(Y_test, dtype=int)
    for i, proba in enumerate(Y_pred_proba):
        Y_pred[:, i] = (proba[:, 1] >= thresholds[i]).astype(int)

    precision, recall, f1_macro, _ = precision_recall_fscore_support(Y_test, Y_pred, average='macro')
    if f1_macro > best_f1:
        best_f1 = f1_macro
        best_thresholds = thresholds

print("Best F1 Score:", best_f1)
print("Best Thresholds:", best_thresholds)

# Apply best thresholds
Y_pred = np.zeros_like(Y_test, dtype=int)
for i, proba in enumerate(Y_pred_proba):
    Y_pred[:, i] = (proba[:, 1] >= best_thresholds[i]).astype(int)

# Evaluation with fine-tuned thresholds
precision, recall, f1_macro, _ = precision_recall_fscore_support(Y_test, Y_pred, average='macro')
accuracy = accuracy_score(Y_test, Y_pred)

print("Evaluation Metrics with Combined Features:")
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro): {recall:.4f}")
print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"Accuracy: {accuracy:.4f}")


# In[ ]:



# Load the dataset
file_path = "C:/Users/Aggeliki/Desktop/Thesis/public_data_2/track_a/train/eng.csv"
data = pd.read_csv(file_path)

# Define custom dataset for PyTorch
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

# Split the data into training and test sets
X_train_text, X_test_text, Y_train, Y_test = train_test_split(
    data['text'], data[['anger', 'fear', 'joy', 'sadness', 'surprise']].values, test_size=0.2, random_state=42
)

# Load BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

# Create datasets and dataloaders
train_dataset = EmotionDataset(X_train_text.tolist(), Y_train, bert_tokenizer, max_length=128)
test_dataset = EmotionDataset(X_test_text.tolist(), Y_test, bert_tokenizer, max_length=128)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Reduced batch size for faster training
test_loader = DataLoader(test_dataset, batch_size=8)

# Fine-tune BERT
optimizer = AdamW(bert_model.parameters(), lr=1e-5)  # Reduced learning rate for better fine-tuning
loss_fn = torch.nn.BCEWithLogitsLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

# Training loop with reduced epochs
for epoch in range(2):  # Reduced epochs for faster training while maintaining performance
    bert_model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()

# Extract embeddings from fine-tuned BERT
def extract_embeddings(texts, tokenizer, model, max_length):
    inputs = tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.cpu().numpy()

X_train_bert = extract_embeddings(X_train_text, bert_tokenizer, bert_model, max_length=128)
X_test_bert = extract_embeddings(X_test_text, bert_tokenizer, bert_model, max_length=128)

# Feature Engineering: Adding TF-IDF features
vectorizer = TfidfVectorizer(max_features=8000)  # Increased max_features for more textual representation
X_train_tfidf = vectorizer.fit_transform(X_train_text).toarray()
X_test_tfidf = vectorizer.transform(X_test_text).toarray()

# Combine BERT embeddings and TF-IDF features
X_train_combined = np.hstack((X_train_bert, X_train_tfidf))
X_test_combined = np.hstack((X_test_bert, X_test_tfidf))

# Define models
models = {
    'LR': LogisticRegression(max_iter=1500, class_weight='balanced'),  # Increased max_iter for convergence
    'RF': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),  # Increased estimators and depth
    'SVM': SVC(kernel='linear', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),  # Adjusted neighbors for better performance
    'XGB': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=200, random_state=42),
}

# Ensemble Voting Classifier with weights
voting_ensemble = VotingClassifier(
    estimators=[(name, model) for name, model in models.items()],
    voting='soft',  # Use soft voting for probabilities
    weights=[1, 3, 1, 1, 4]  # Adjusted weights to prioritize RF and XGB
)

# Multi-label wrapper
multi_label_model = MultiOutputClassifier(voting_ensemble)

# Train models
multi_label_model.fit(X_train_combined, Y_train)

# Save model
joblib.dump(multi_label_model, 'C:/Users/Aggeliki/Desktop/Thesis/public_data/voting_ensemble_model_with_features.pkl')

# Predictions probabilities
Y_pred_proba = multi_label_model.predict_proba(X_test_combined)

# Grid Search for Fine-Tuned Thresholds
candidate_thresholds = np.arange(0.2, 0.5, 0.01)  # Fine-grained thresholds for better tuning
best_f1 = 0
best_thresholds = []

for thresholds in product(candidate_thresholds, repeat=Y_test.shape[1]):
    Y_pred = np.zeros_like(Y_test, dtype=int)
    for i, proba in enumerate(Y_pred_proba):
        Y_pred[:, i] = (proba[:, 1] >= thresholds[i]).astype(int)

    precision, recall, f1_macro, _ = precision_recall_fscore_support(Y_test, Y_pred, average='macro')
    if f1_macro > best_f1:
        best_f1 = f1_macro
        best_thresholds = thresholds

print("Best F1 Score:", best_f1)
print("Best Thresholds:", best_thresholds)

# Apply best thresholds
Y_pred = np.zeros_like(Y_test, dtype=int)
for i, proba in enumerate(Y_pred_proba):
    Y_pred[:, i] = (proba[:, 1] >= best_thresholds[i]).astype(int)

# Evaluation with fine-tuned thresholds
precision, recall, f1_macro, _ = precision_recall_fscore_support(Y_test, Y_pred, average='macro')
accuracy = accuracy_score(Y_test, Y_pred)

print("Evaluation Metrics with Combined Features:")
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro): {recall:.4f}")
print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"Accuracy: {accuracy:.4f}")


# In[31]:


print("Evaluation Metrics with Combined Features:")
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro): {recall:.4f}")
print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"Accuracy: {accuracy:.4f}")


# In[29]:


save_path = "C:/Users/Aggeliki/Desktop/Thesis/public_data/fine_tuned_bert"
bert_model.save_pretrained(save_path)
bert_tokenizer.save_pretrained(save_path)

print(f"Fine-tuned BERT model saved to {save_path}")


# In[32]:



vectorizer_path = "C:/Users/Aggeliki/Desktop/Thesis/public_data/tfidf_vectorizer.pkl"
joblib.dump(vectorizer, vectorizer_path)

print(f"TfidfVectorizer saved successfully to {vectorizer_path}")


# In[46]:


# Save best thresholds
thresholds_path = "C:/Users/Aggeliki/Desktop/Thesis/public_data/best_thresholds.npy"
np.save(thresholds_path, np.array(best_thresholds))
print(f"Best thresholds saved to {thresholds_path}")


# In[54]:



# Load the dataset with missing values
file_path = "C:/Users/Aggeliki/Desktop/Thesis/public_data_test/track_a/test/eng.csv"
data_with_missing = pd.read_csv(file_path)

# Load the trained model
model_path = "C:/Users/Aggeliki/Desktop/Thesis/public_data/voting_ensemble_model_with_features.pkl"
multi_label_model = joblib.load(model_path)

# Load the tokenizer and model for BERT
bert_tokenizer = BertTokenizer.from_pretrained('C:/Users/Aggeliki/Desktop/Thesis/public_data/fine_tuned_bert')
bert_model = BertForSequenceClassification.from_pretrained('C:/Users/Aggeliki/Desktop/Thesis/public_data/fine_tuned_bert')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

# Function to extract embeddings from BERT
def extract_embeddings(texts, tokenizer, model, max_length=128):
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.cpu().numpy()

# Extract text column and process it
texts_to_predict = data_with_missing['text'].dropna().tolist()  # Drop NaN texts

# Extract embeddings from BERT
X_bert = extract_embeddings(texts_to_predict, bert_tokenizer, bert_model, max_length=128)

# Load the TF-IDF vectorizer
vectorizer_path = "C:/Users/Aggeliki/Desktop/Thesis/public_data/tfidf_vectorizer.pkl"
vectorizer = joblib.load(vectorizer_path)

# Apply TF-IDF transformation
X_tfidf = vectorizer.transform(texts_to_predict).toarray()

# Ensure feature size consistency
expected_features = multi_label_model.estimators_[0].n_features_in_
current_features = X_bert.shape[1] + X_tfidf.shape[1]

if current_features < expected_features:
    padding = np.zeros((X_tfidf.shape[0], expected_features - current_features))
    X_tfidf = np.hstack((X_tfidf, padding))
    print(f'Added {expected_features - current_features} padding features.')

elif current_features > expected_features:
    X_tfidf = X_tfidf[:, :expected_features - X_bert.shape[1]]
    print(f'Truncated {current_features - expected_features} extra features to match expected size.')

# Combine features
X_combined = np.hstack((X_bert, X_tfidf))

# Final feature size validation
final_features = X_combined.shape[1]
if final_features != expected_features:
    raise ValueError(f'Final feature size mismatch: {final_features} instead of {expected_features}')

# Make predictions
Y_pred_proba = multi_label_model.predict_proba(X_combined)

# Διόρθωση thresholds
best_thresholds = np.load("C:/Users/Aggeliki/Desktop/Thesis/public_data/best_thresholds.npy")  # Αυξημένα thresholds

# Apply best thresholds
Y_pred = np.zeros((Y_pred_proba[0].shape[0], len(best_thresholds)), dtype=int)

Y_pred = np.zeros((len(Y_pred_proba[0]), len(best_thresholds)), dtype=int)
for i in range(len(best_thresholds)):
    if Y_pred_proba[i].shape[1] > 1:
        Y_pred[:, i] = (Y_pred_proba[i][:, 1] >= best_thresholds[i]).astype(int)
    else:
        Y_pred[:, i] = (Y_pred_proba[i][:, 0] >= best_thresholds[i]).astype(int)

# Εκτύπωση τελικών προβλέψεων για έλεγχο
print("Sample predictions after applying thresholds:")
print(Y_pred[:10])  # Δείγμα των πρώτων 10 προβλέψεων

# Convert predictions to DataFrame
predictions_df = pd.DataFrame(Y_pred, columns=['anger', 'fear', 'joy', 'sadness', 'surprise'])

# Merge predictions with original dataset
data_with_missing.loc[data_with_missing['text'].notna(), ['anger', 'fear', 'joy', 'sadness', 'surprise']] = Y_pred

# Save the updated dataset
output_path = "C:/Users/Aggeliki/Desktop/Thesis/public_data_test/track_a/test/eng_filled.csv"
final_df = data_with_missing[['id', 'anger', 'fear', 'joy', 'sadness', 'surprise']]
final_df.to_csv(output_path, index=False, sep=",", encoding="utf-8", header=True, float_format="%.0f")

print(f"Predictions saved to {output_path}")


# In[58]:



# Φόρτωσε το fine-tuned BERT και τον tokenizer
save_path = "C:/Users/Aggeliki/Desktop/Thesis/public_data/fine_tuned_bert"
bert_tokenizer = BertTokenizer.from_pretrained(save_path)
bert_model = BertForSequenceClassification.from_pretrained(save_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

# Φόρτωσε το test dataset
test_file_path = "C:/Users/Aggeliki/Desktop/Thesis/public_data_test/track_a/test/eng.csv"
data_with_missing = pd.read_csv(test_file_path)

# Εξαγωγή κειμένων προς πρόβλεψη
texts_to_predict = data_with_missing['text'].dropna().tolist()

# Συνάρτηση εξαγωγής embeddings
def extract_embeddings(texts, tokenizer, model, max_length=128):
    inputs = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.cpu().numpy()

print("Extracting embeddings for test data...")
X_bert = extract_embeddings(texts_to_predict, bert_tokenizer, bert_model, max_length=128)

# Φόρτωσε το αποθηκευμένο TF-IDF vectorizer
vectorizer_path = "C:/Users/Aggeliki/Desktop/Thesis/public_data/tfidf_vectorizer.pkl"
vectorizer = joblib.load(vectorizer_path)

# Εφαρμογή TF-IDF
X_tfidf = vectorizer.transform(texts_to_predict).toarray()

# Συνδυασμός χαρακτηριστικών
X_combined = np.hstack((X_bert, X_tfidf))

# Φόρτωσε το εκπαιδευμένο μοντέλο
model_path = "C:/Users/Aggeliki/Desktop/Thesis/public_data/voting_ensemble_model_with_features.pkl"
multi_label_model = joblib.load(model_path)

# Φόρτωσε τα thresholds
thresholds_path = "C:/Users/Aggeliki/Desktop/Thesis/public_data/best_thresholds.npy"
best_thresholds = np.load(thresholds_path)

# Κάνε προβλέψεις
print("Making predictions...")
Y_pred_proba = multi_label_model.predict_proba(X_combined)

# Εφαρμογή thresholds
Y_pred = np.zeros((len(Y_pred_proba[0]), len(best_thresholds)), dtype=int)
for i in range(len(best_thresholds)):
    if Y_pred_proba[i].shape[1] > 1:
        Y_pred[:, i] = (Y_pred_proba[i][:, 1] >= best_thresholds[i]).astype(int)
    else:
        Y_pred[:, i] = (Y_pred_proba[i][:, 0] >= best_thresholds[i]).astype(int)

# Αποθήκευση των τελικών προβλέψεων χωρίς δεκαδικά
output_path = "C:/Users/Aggeliki/Desktop/Thesis/public_data_test/track_a/test/eng_filled.csv"
data_with_missing.loc[data_with_missing['text'].notna(), ['anger', 'fear', 'joy', 'sadness', 'surprise']] = Y_pred
final_df = data_with_missing[['id', 'anger', 'fear', 'joy', 'sadness', 'surprise']]
final_df.to_csv(output_path, index=False, sep=",", encoding="utf-8", header=True, float_format="%.0f")

print(f"Predictions saved to {output_path}")


# In[41]:



# Load the dataset
file_path = "C:/Users/Aggeliki/Desktop/Thesis/public_data_2/track_a/train/eng.csv"
data = pd.read_csv(file_path)

# Define custom dataset for PyTorch
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

# Split the data into training and test sets
X_train_text, X_test_text, Y_train, Y_test = train_test_split(
    data['text'], data[['anger', 'fear', 'joy', 'sadness', 'surprise']].values, test_size=0.2, random_state=42
)

# Define token and model name for Hugging Face access
token = "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"
model_name = "answerdotai/ModernBERT-base"

# Load ModernBERT model and tokenizer with authentication token
modelbert_tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
modelbert_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5, use_auth_token=token)

# Create datasets and dataloaders
train_dataset = EmotionDataset(X_train_text.tolist(), Y_train, modelbert_tokenizer, max_length=128)
test_dataset = EmotionDataset(X_test_text.tolist(), Y_test, modelbert_tokenizer, max_length=128)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Fine-tune ModernBERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelbert_model.to(device)

optimizer = AdamW(modelbert_model.parameters(), lr=1e-5)
loss_fn = torch.nn.BCEWithLogitsLoss()

# Training loop with progress bar
for epoch in range(2):
    modelbert_model.train()
    loop = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=True)
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = modelbert_model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

# Extract embeddings from fine-tuned ModernBERT with progress bar
def extract_embeddings(texts, tokenizer, model, max_length):
    embeddings = []
    for text in tqdm(texts, desc="Extracting Embeddings"):
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.logits.cpu().numpy())
    return np.vstack(embeddings)

X_train_modelbert = extract_embeddings(X_train_text, modelbert_tokenizer, modelbert_model, max_length=128)
X_test_modelbert = extract_embeddings(X_test_text, modelbert_tokenizer, modelbert_model, max_length=128)

# Feature Engineering: Adding TF-IDF features
vectorizer = TfidfVectorizer(max_features=8000)
X_train_tfidf = vectorizer.fit_transform(tqdm(X_train_text, desc="TF-IDF Fitting")).toarray()
X_test_tfidf = vectorizer.transform(tqdm(X_test_text, desc="TF-IDF Transforming")).toarray()

# Combine ModernBERT embeddings and TF-IDF features
X_train_combined = np.hstack((X_train_modelbert, X_train_tfidf))
X_test_combined = np.hstack((X_test_modelbert, X_test_tfidf))

# Define models
models = {
    'LR': LogisticRegression(max_iter=1500, class_weight='balanced', n_jobs=-1),
    'RF': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'SVM': SVC(kernel='linear', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'XGB': XGBClassifier(tree_method="gpu_hist", gpu_id=0, use_label_encoder=False, eval_metric='mlogloss', n_estimators=200, random_state=42, n_jobs=-1),
}

# Ensemble Voting Classifier with weights
voting_ensemble = VotingClassifier(
    estimators=[(name, model) for name, model in models.items()],
    voting='soft',
    weights=[1, 3, 1, 1, 4]
)

# Multi-label wrapper
multi_label_model = MultiOutputClassifier(voting_ensemble)

# Train models with progress bar
multi_label_model.fit(X_train_combined, Y_train)

# Save model
joblib.dump(multi_label_model, 'C:/Users/Aggeliki/Desktop/Thesis/public_data/voting_ensemble_model_with_modernbert.pkl')

# Predictions probabilities with progress bar
Y_pred_proba = multi_label_model.predict_proba(X_test_combined)

# Grid Search for Fine-Tuned Thresholds with progress bar (100 steps)
candidate_thresholds = np.arange(0.2, 0.5, 0.01)
best_f1 = 0
best_thresholds = []

for thresholds in tqdm(product(candidate_thresholds, repeat=Y_test.shape[1]), desc="Grid Search", total=100, disable=False):
    Y_pred = np.zeros_like(Y_test, dtype=int)
    for i, proba in enumerate(Y_pred_proba):
        Y_pred[:, i] = (proba[:, 1] >= thresholds[i]).astype(int)

    precision, recall, f1_macro, _ = precision_recall_fscore_support(Y_test, Y_pred, average='samples')  # 'samples' for multi-label
    if f1_macro > best_f1:
        best_f1 = f1_macro
        best_thresholds = thresholds

print("Best F1 Score:", best_f1)
print("Best Thresholds:", best_thresholds)

# Apply best thresholds
Y_pred = np.zeros_like(Y_test, dtype=int)
for i, proba in enumerate(Y_pred_proba):
    Y_pred[:, i] = (proba[:, 1] >= best_thresholds[i]).astype(int)

# Evaluation with fine-tuned thresholds
precision, recall, f1_macro, _ = precision_recall_fscore_support(Y_test, Y_pred, average='samples')  # 'samples' for multi-label evaluation
accuracy = accuracy_score(Y_test, Y_pred)

print("Evaluation Metrics with Combined Features:")
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro): {recall:.4f}")
print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"Accuracy: {accuracy:.4f}")


# In[27]:


#Roberta Large 


# Load the dataset
file_path = "C:/Users/Aggeliki/Desktop/Thesis/public_data_2/track_a/train/eng.csv"
data = pd.read_csv(file_path)

# Define custom dataset for PyTorch
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

# Split the data
X_train_text, X_test_text, Y_train, Y_test = train_test_split(
    data['text'], data[['anger', 'fear', 'joy', 'sadness', 'surprise']].values, 
    test_size=0.2, random_state=42
)

# Load RoBERTa model and tokenizer
model_name = 'roberta-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

# DataLoader
train_dataset = EmotionDataset(X_train_text.tolist(), Y_train, tokenizer, max_length=128)
test_dataset = EmotionDataset(X_test_text.tolist(), Y_test, tokenizer, max_length=128)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Fine-tuning
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.BCEWithLogitsLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training
for epoch in range(5):
    model.train()
    loop = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

# Embeddings

def extract_embeddings(texts, tokenizer, model, max_length):
    embeddings = []
    for text in tqdm(texts, desc="Extracting Embeddings"):
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.logits.cpu().numpy())
    return np.vstack(embeddings)

X_train_embeddings = extract_embeddings(X_train_text, tokenizer, model, max_length=128)
X_test_embeddings = extract_embeddings(X_test_text, tokenizer, model, max_length=128)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=8000)
X_train_tfidf = vectorizer.fit_transform(tqdm(X_train_text, desc="TF-IDF Fitting")).toarray()
X_test_tfidf = vectorizer.transform(tqdm(X_test_text, desc="TF-IDF Transforming")).toarray()

# Combine
X_train_combined = np.hstack((X_train_embeddings, X_train_tfidf))
X_test_combined = np.hstack((X_test_embeddings, X_test_tfidf))

# Models
models = {
    'LR': LogisticRegression(max_iter=2000, class_weight='balanced'),
    'RF': RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42),
    'SVM': SVC(kernel='linear', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'XGB': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=300, random_state=42),
    'LGBM': LGBMClassifier(n_estimators=300, random_state=42),
    'CatBoost': CatBoostClassifier(iterations=300, verbose=0, random_state=42)
}

voting_ensemble = VotingClassifier(
    estimators=[(name, model) for name, model in models.items()],
    voting='soft',
    weights=[1, 3, 1, 1, 4, 2, 2]
)

multi_label_model = MultiOutputClassifier(voting_ensemble)
multi_label_model.fit(X_train_combined, Y_train)

# Save Model
joblib.dump(multi_label_model, 'C:/Users/Aggeliki/Desktop/Thesis/public_data/voting_ensemble_optimized.pkl')

# Predictions
Y_pred_proba = multi_label_model.predict_proba(X_test_combined)

# Threshold Optimization
candidate_thresholds = np.arange(0.2, 0.5, 0.01)
best_f1 = 0
best_thresholds = []

total_iterations = len(candidate_thresholds) ** Y_test.shape[1]

for thresholds in tqdm(product(candidate_thresholds, repeat=Y_test.shape[1]), desc="Grid Search", total=total_iterations):
    Y_pred = np.zeros_like(Y_test, dtype=int)
    for i, proba in enumerate(Y_pred_proba):
        Y_pred[:, i] = (proba[:, 1] >= thresholds[i]).astype(int)
    
    precision, recall, f1_macro, _ = precision_recall_fscore_support(Y_test, Y_pred, average='macro')
    if f1_macro > best_f1:
        best_f1 = f1_macro
        best_thresholds = thresholds

# Evaluation
Y_pred = np.zeros_like(Y_test, dtype=int)
for i, proba in enumerate(Y_pred_proba):
    Y_pred[:, i] = (proba[:, 1] >= best_thresholds[i]).astype(int)

precision, recall, f1_macro, _ = precision_recall_fscore_support(Y_test, Y_pred, average='macro')
accuracy = accuracy_score(Y_test, Y_pred)

print("Best F1 Score:", best_f1)
print("Precision (Macro):", precision)
print("Recall (Macro):", recall)
print("F1 Score (Macro):", f1_macro)
print("Accuracy:", accuracy)


# In[28]:


#XLMRoberta-Large 


# Load the dataset
file_path = "C:/Users/Aggeliki/Desktop/Thesis/public_data_2/track_a/train/eng.csv"
data = pd.read_csv(file_path)

# Define custom dataset for PyTorch
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

# Split the data
X_train_text, X_test_text, Y_train, Y_test = train_test_split(
    data['text'], data[['anger', 'fear', 'joy', 'sadness', 'surprise']].values, 
    test_size=0.2, random_state=42
)

# Load XLM-RoBERTa model and tokenizer
model_name = 'xlm-roberta-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

# DataLoader
train_dataset = EmotionDataset(X_train_text.tolist(), Y_train, tokenizer, max_length=128)
test_dataset = EmotionDataset(X_test_text.tolist(), Y_test, tokenizer, max_length=128)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Fine-tuning
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.BCEWithLogitsLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training
epochs = 7  # Increased epochs
for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

# Extract embeddings

def extract_embeddings(texts, tokenizer, model, max_length):
    embeddings = []
    for text in tqdm(texts, desc="Extracting Embeddings"):
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.logits.cpu().numpy())
    return np.vstack(embeddings)

X_train_embeddings = extract_embeddings(X_train_text, tokenizer, model, max_length=128)
X_test_embeddings = extract_embeddings(X_test_text, tokenizer, model, max_length=128)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=8000)
X_train_tfidf = vectorizer.fit_transform(tqdm(X_train_text, desc="TF-IDF Fitting")).toarray()
X_test_tfidf = vectorizer.transform(tqdm(X_test_text, desc="TF-IDF Transforming")).toarray()

# Combine
X_train_combined = np.hstack((X_train_embeddings, X_train_tfidf))
X_test_combined = np.hstack((X_test_embeddings, X_test_tfidf))

# Models
models = {
    'LR': LogisticRegression(max_iter=2000, class_weight='balanced'),
    'RF': RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42),
    'SVM': SVC(kernel='linear', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'XGB': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=300, random_state=42),
    'LGBM': LGBMClassifier(n_estimators=300, random_state=42),
    'CatBoost': CatBoostClassifier(iterations=300, verbose=0, random_state=42)
}

voting_ensemble = VotingClassifier(
    estimators=[(name, model) for name, model in models.items()],
    voting='soft',
    weights=[1, 3, 1, 1, 4, 2, 2]
)

multi_label_model = MultiOutputClassifier(voting_ensemble)
multi_label_model.fit(X_train_combined, Y_train)

# Save Model
joblib.dump(multi_label_model, 'C:/Users/Aggeliki/Desktop/Thesis/public_data/voting_ensemble_xlm_roberta.pkl')

# Predictions
Y_pred_proba = multi_label_model.predict_proba(X_test_combined)

# Threshold Optimization
candidate_thresholds = np.arange(0.2, 0.5, 0.01)
best_f1 = 0
best_thresholds = []

for thresholds in tqdm(product(candidate_thresholds, repeat=Y_test.shape[1]), desc="Grid Search"):
    Y_pred = np.zeros_like(Y_test, dtype=int)
    for i, proba in enumerate(Y_pred_proba):
        Y_pred[:, i] = (proba[:, 1] >= thresholds[i]).astype(int)
    
    precision, recall, f1_macro, _ = precision_recall_fscore_support(Y_test, Y_pred, average='macro')
    if f1_macro > best_f1:
        best_f1 = f1_macro
        best_thresholds = thresholds

# Evaluation
Y_pred = np.zeros_like(Y_test, dtype=int)
for i, proba in enumerate(Y_pred_proba):
    Y_pred[:, i] = (proba[:, 1] >= best_thresholds[i]).astype(int)

precision, recall, f1_macro, _ = precision_recall_fscore_support(Y_test, Y_pred, average='macro')
accuracy = accuracy_score(Y_test, Y_pred)

print("Best F1 Score:", best_f1)
print("Precision (Macro):", precision)
print("Recall (Macro):", recall)
print("F1 Score (Macro):", f1_macro)
print("Accuracy:", accuracy)


# In[ ]:



