# Supervised Models - Cleaned with English comments
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
import joblib  
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt 
import nltk
import numpy as np
import numpy as np 
import optuna
import pandas as pd
import re
import seaborn as sns
import seaborn as sns 
import warnings

from pathlib import Path
base_dir = Path(__file__).parent
figure_dir = base_dir / 'figures'
figure_dir.mkdir(exist_ok=True)
results_dir = base_dir / 'results'
results_dir.mkdir(exist_ok=True)

#!/usr/bin/env python
# coding: utf-8

# In[27]:




# In[28]:


#load the data from csv file to pandas dataframe 
training_data= pd.read_csv("C:/Users/Aggeliki/Desktop/Thesis/public_data_2/track_a/train/eng.csv")


# In[29]:


#print the first 5 rows of the dataframe 
training_data.head()


# In[30]:


#How many times every emotion appears

# Sum of each emotion
emotion_counts = training_data[['anger', 'fear', 'joy', 'sadness', 'surprise']].sum()

# Create the bar plot
plt.figure(figsize=(8, 6))
emotion_counts.plot(kind='bar', color=['red', 'orange', 'yellow', 'blue', 'green'])

plt.title('Emotion Count')
plt.xlabel('Emotion')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig(figure_dir / 'plot_01.png')
plt.close()


# In[54]:


# Βρίσκουμε την πλειοψηφία των labels
majority_emotion = emotion_counts.idxmax()
majority_count = emotion_counts.max()

print(f"The majority emotion is '{majority_emotion}' with {majority_count} occurrences.")

# Υπολογισμός ποσοστών κάθε συναισθήματος
emotion_percentages = (emotion_counts / emotion_counts.sum()) * 100

# Εκτύπωση αποτελεσμάτων
print(emotion_percentages)


# In[31]:


# Count the number of rows in the dataframe
total_texts = training_data.shape[0]

print(f'Total number of texts: {total_texts}')


# In[32]:


#How many times appears the emotion 'Anger' (1) or not appear (0)
anger_appears = training_data['anger'].sum()  
anger_not_appears = training_data.shape[0] - anger_appears  

print(f"Anger appears: {anger_appears}")
print(f"Anger does not appear: {anger_not_appears}")

# Data for the bar plot
anger_data = pd.Series([anger_appears, anger_not_appears], index=['1', '0'])

# Create the bar plot
plt.figure(figsize=(6, 4))
anger_data.plot(kind='bar', color=['green', 'red'])

plt.title('Anger Appearance in Texts')
plt.xlabel('Anger Status')
plt.ylabel('Count')

plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(figure_dir / 'plot_02.png')
plt.close()


# In[33]:


#How many times appears the emotion 'Fear' (1) or not appear (0)
fear_appears = training_data['fear'].sum()  
fear_not_appears = training_data.shape[0] - fear_appears  

print(f"Fear appears: {fear_appears}")
print(f"Fear does not appear: {fear_not_appears}")

# Data for the bar plot
fear_data = pd.Series([fear_appears, fear_not_appears], index=['1', '0'])

# Create the bar plot
plt.figure(figsize=(6, 4))
fear_data.plot(kind='bar', color=['green', 'red'])

plt.title('Fear Appearance in Texts')
plt.xlabel('Fear Status')
plt.ylabel('Count')

plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(figure_dir / 'plot_03.png')
plt.close()


# In[34]:


#How many times the emotion 'Joy' appears (1) or does not appear (0)
joy_appears = training_data['joy'].sum()  
joy_not_appears = training_data.shape[0] - joy_appears  

print(f"Joy appears: {joy_appears}")
print(f"Joy does not appear: {joy_not_appears}")

# Data for the bar plot
joy_data = pd.Series([joy_appears, joy_not_appears], index=['1', '0'])

# Create the bar plot
plt.figure(figsize=(6, 4))
joy_data.plot(kind='bar', color=['green', 'red'])

plt.title('Joy Appearance in Texts')
plt.xlabel('Joy Status')
plt.ylabel('Count')

plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(figure_dir / 'plot_04.png')
plt.close()


# In[35]:


#How many times the emotion 'Sadness' appears (1) or does not appear (0)
sadness_appears = training_data['sadness'].sum()  
sadness_not_appears = training_data.shape[0] - sadness_appears  

print(f"Sadness appears: {sadness_appears}")
print(f"Sadness does not appear: {sadness_not_appears}")

# Data for the bar plot
sadness_data = pd.Series([sadness_appears, sadness_not_appears], index=['1', '0'])

# Create the bar plot
plt.figure(figsize=(6, 4))
sadness_data.plot(kind='bar', color=['green', 'red'])

plt.title('Sadness Appearance in Texts')
plt.xlabel('Sadness Status')
plt.ylabel('Count')

plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(figure_dir / 'plot_05.png')
plt.close()


# In[36]:


#How many times the emotion 'Surprise' appears (1) or does not appear (0)
surprise_appears = training_data['surprise'].sum()  
surprise_not_appears = training_data.shape[0] - surprise_appears  

print(f"Surprise appears: {surprise_appears}")
print(f"Surprise does not appear: {surprise_not_appears}")

# Data for the bar plot
surprise_data = pd.Series([surprise_appears, surprise_not_appears], index=['1', '0'])

# Create the bar plot
plt.figure(figsize=(6, 4))
surprise_data.plot(kind='bar', color=['green', 'red'])

plt.title('Surprise Appearance in Texts')
plt.xlabel('Surprise Status')
plt.ylabel('Count')

plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(figure_dir / 'plot_06.png')
plt.close()


# In[37]:


#Barplot with all the emotions appears 
#Barplot with all the emotions appears 


data = pd.DataFrame({
    'anger': [anger_appears],
    'fear': [fear_appears],
    'joy': [joy_appears],
    'sadness': [sadness_appears],
    'surprise': [surprise_appears]
}, index=['Appears'])

# Set the Seaborn color palette
colors = ['#FF0000',  
          '#FFA500',  
          '#008000',  
          '#0000FF',  
          '#FFFF00']  

# Create the bar plot
plt.figure(figsize=(8, 6))
data.plot(kind='bar', color=colors[0:5])

# Add titles and labels
plt.title('Emotions Appearance in Texts')
plt.xlabel('Emotion Status')
plt.ylabel('Count')

# Rotate x-ticks
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(figure_dir / 'plot_07.png')
plt.close()

#Second table 

# Sample data
data = pd.DataFrame({
    'anger': [anger_appears],
    'fear': [fear_appears],
    'joy': [joy_appears],
    'sadness': [sadness_appears],
    'surprise': [surprise_appears]
}, index=['Appears'])

# Χρώματα pastel με απαλή απόδοση
colors = ['#D98880',  # Απαλό κόκκινο
          '#F5CBA7',  # Απαλό πορτοκαλί
          '#AED6F1',  # Απαλό μπλε
          '#D7BDE2',  # Απαλό μοβ
          '#A9DFBF']  # Απαλό πράσινο

# Δημιουργία διαγράμματος
plt.figure(figsize=(8, 6))
data.plot(kind='bar', color=colors,  alpha=0.8)  # Alpha για πιο απαλή αίσθηση

# Προσθήκη τίτλου και ετικετών
plt.title('Emotions Appearance in Texts', fontsize=14)
plt.xlabel('Emotion Status', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Περιστροφή x-ticks
plt.xticks(rotation=0)

# Προσαρμογή layout
plt.tight_layout()

# Εμφάνιση γραφήματος
plt.savefig(figure_dir / 'plot_08.png')
plt.close()


# In[38]:


# Count how many rows have null values for all emotion columns
null_emotion_count = training_data.isnull().all(axis=1).sum()

print(f"Number of texts with null emotions: {null_emotion_count}")


# In[39]:


# Select only the numeric columns 
emotion_columns = training_data.select_dtypes(include=['number'])

# Count how many rows have all the emotions 0 
only_zeros_count = (emotion_columns.sum(axis=1) == 0).sum()

# Count how many rows have all the emotions 1 
only_ones_count = (emotion_columns.sum(axis=1) == emotion_columns.shape[1]).sum()

print(f"Number of texts with all the emotions 0 : {only_zeros_count}")
print(f"Number of texts with all the emotions 1 : {only_ones_count}")


# In[40]:


# Get the rows with all the emotions 1
only_ones_rows = training_data[emotion_columns.sum(axis=1) == emotion_columns.shape[1]]
print("\nRows with all the emotions 1:")
print(only_ones_rows.head(2))  

# Get the first 5 rows with all the emotions 0 
only_zeros_rows = training_data[emotion_columns.sum(axis=1) == 0]
print("\nFirst 5 rows with all the emotions 0 :")
print(only_zeros_rows.head())  


# In[41]:


# Prepare data for the bar plot
counts = pd.Series([only_zeros_count, only_ones_count], index=['Only 0', 'Only 1'])

# Create the bar plot
plt.figure(figsize=(6, 4))
counts.plot(kind='bar', color=['blue', 'orange'])

plt.title('Count of Texts with all the emotions 0 and all the emotions 1')
plt.xlabel('Emotion Status')
plt.ylabel('Count')

plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(figure_dir / 'plot_09.png')
plt.close()


# In[42]:


# Select only the numeric columns 
emotion_columns = training_data.select_dtypes(include=['number'])

# Count how many rows have exactly one '1'
only_one_emotion_rows = (emotion_columns.sum(axis=1) == 1) 
only_one_emotion_count = only_one_emotion_rows.sum() 

print(f"Number of texts with exactly one emotion (only one '1'): {only_one_emotion_count}")

# Count how many rows have more '1' emotion
two_or_more_emotion_rows = (emotion_columns.sum(axis=1) >= 2) 
two_or_more_emotion_count = two_or_more_emotion_rows.sum()  

print(f"Number of texts with more emotions: {two_or_more_emotion_count}")


# Data for the bar plot
labels = ['Exactly One Emotion', 'More Emotions']
counts = [only_one_emotion_count, two_or_more_emotion_count]

# Create the bar plot
plt.figure(figsize=(8, 5))
plt.bar(labels, counts, color=['blue', 'orange'])

plt.title('Comparison of Texts by Emotion Count')
plt.xlabel('Emotion Status')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig(figure_dir / 'plot_10.png')
plt.close()


#Second Diagram 
# Δεδομένα για το bar plot
labels = ['Exactly One Emotion', 'More Emotions']
counts = [only_one_emotion_count, two_or_more_emotion_count]

# Νέα pastel χρωματική παλέτα με ακόμα πιο απαλά κόκκινα & πράσινα
colors = ['#F4A7A3',  # Πολύ απαλό κόκκινο
          '#B7E1CD']  # Πολύ απαλό πράσινο

# Δημιουργία του bar plot με τις νέες pastel αποχρώσεις
plt.figure(figsize=(8, 5))
plt.bar(labels, counts, color=colors, alpha=0.8)  # Ακόμα πιο απαλή εμφάνιση

# Προσθήκη τίτλου και labels
plt.title('Comparison of Texts by Emotion Count', fontsize=14)
plt.xlabel('Emotion Status', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Βελτίωση της διάταξης
plt.tight_layout()

# Εμφάνιση γραφήματος
plt.savefig(figure_dir / 'plot_11.png')
plt.close()


# In[43]:


# Define which emotion columns correspond to each sentiment
positive_emotions = ['joy'] 
negative_emotions = ['anger', 'sadness', 'fear']  
neutral_emotions = ['surprise']  

def classify_sentiment(row):
    is_positive = any(row[positive_emotions])
    is_negative = any(row[negative_emotions])
    is_neutral = any(row[neutral_emotions])
    
    if is_positive and not is_negative:
        return 'Positive'
    elif is_negative and not is_positive:
        return 'Negative'
    elif is_positive and is_negative:
        return 'Mixed'  # Text has both positive and negative emotions
    elif is_neutral:
        return 'Neutral'
    else:
        return 'No Emotion'

# Create a new 'Sentiment' column
training_data['sentiment'] = training_data.apply(classify_sentiment, axis=1)

# Count how many texts are in each sentiment category
sentiment_counts = training_data['sentiment'].value_counts()
 
print(sentiment_counts)

# Data for the bar plot
labels = sentiment_counts.index
counts = sentiment_counts.values

# Create the bar plot
plt.figure(figsize=(8, 5))
plt.bar(labels, counts, color=['red', 'green', 'gray', 'orange', 'blue'])

plt.title('Count of Texts by Sentiment Based on Emotions')
plt.xlabel('Sentiment')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig(figure_dir / 'plot_12.png')
plt.close()


# In[44]:




# In[45]:


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'\d+', '', text)  # Removing digits
    text = re.sub(r'[^\w\s]', '', text)  # Removing punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])  # Removing stopwords
    return text


# In[46]:


X_clean = training_data['text'].apply(preprocess_text)

# Define input (text) and output (emotion labels)
X = X_clean
Y = training_data[['anger', 'fear', 'joy', 'sadness', 'surprise']]


# In[47]:


# Split into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[48]:


xtrain = X_train.shape[0]
xtest = X_test.shape[0]
ytrain = Y_train.shape[0]
ytest = Y_test.shape[0]

totalx=X.shape[0]
totaly= Y.shape[0]

print(xtrain)
print(xtest)
print(ytrain)
print(ytest)

print(totalx)
print(totaly)


# In[49]:


print(Y_test)


# In[50]:


get_ipython().system('pip install xgboost')


# In[51]:



# Suppress all warnings
warnings.filterwarnings('ignore')

# Define models
models = {
    'LR': LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs'),
    'RF': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'XGB': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

results = []

print('Print only F1 per emotion:')

for i, (model_name, model) in enumerate(models.items(), start=1):
    
    #Initialise CountVectorizer 
    #count_vectorizer = CountVectorizer(max_features=5000)
    #X_train_counts = count_vectorizer.fit_transform(X_train)
    #X_test_counts = count_vectorizer.transform(X_test)
    
    #multi_label_model = MultiOutputClassifier(model)
    #multi_label_model.fit(X_train_counts, Y_train)
    
    #Y_pred = multi_label_model.predict(X_test_counts)
    
    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    joblib.dump(tfidf_vectorizer, 'C:/Users/Aggeliki/Desktop/Thesis/public_data/tfidf_vectorizer.pkl')

    multi_label_model = MultiOutputClassifier(model)
    multi_label_model.fit( X_train_tfidf, Y_train)
    
    joblib.dump(multi_label_model, 'saved_model.pkl')
    
    Y_pred = multi_label_model.predict(X_test_tfidf)
    
    # Macro precision, recall, and f1 score
    precision, recall, f1_macro, _ = precision_recall_fscore_support(Y_test, Y_pred, average='macro')
    
    # Micro precision, recall, and f1 score
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(Y_test, Y_pred, average='micro')
    
    # F1 score per emotion
    f1_per_emotions = f1_score(Y_test, Y_pred, average=None)
    
    #Samples-F1 
    sample_f1_scores = [
        f1_score(Y_test.iloc[i], Y_pred[i], average='weighted') for i in range(len(Y_test))
    ]
    sample_f1 = sum(sample_f1_scores) / len(sample_f1_scores)
    
    # Accuracy score
    accuracy = accuracy_score(Y_test, Y_pred)
    
    results.append({
        'Model': model_name,
        'Precision': precision,
        'Recall': recall,
        'F1 Score (Macro)': f1_macro,
        'F1 Score (Micro)': f1_micro,
        'Accuracy': accuracy,
        'F1 Score per Emotions': f1_per_emotions,
        'Samples-F1': sample_f1
    })
    
    # Print F1 per emotion
    print(f"{model_name}: {f1_per_emotions}")

results_df = pd.DataFrame(results)

print('\nResult Table:')
print(results_df[['Model', 'Precision', 'Recall', 'F1 Score (Macro)','F1 Score (Micro)','Samples-F1', 'Accuracy']])

# Plot overall metrics for each model
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df.melt(id_vars='Model', value_vars=['Precision', 'Recall', 'F1 Score (Macro)','F1 Score (Micro)','Samples-F1', 'Accuracy']),
            x='Model', y='value', hue='variable')
plt.title("Overall Metrics for Each Model")
plt.ylabel("Score")
plt.xlabel("Model")
plt.legend(title="Metric")
plt.savefig(figure_dir / 'plot_13.png')
plt.close()

# Plot the F1 score per emotion for each model
plt.figure(figsize=(12, 8))

# Extract F1 per emotions for each model into a dictionary for plotting
f1_scores_per_emotions= {result['Model']: result['F1 Score per Emotions'] for result in results}

emotion_labels = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']
f1_scores_per_emotions_df = pd.DataFrame(f1_scores_per_emotions).T
f1_scores_per_emotions_df.columns = emotion_labels

custom_colors = ['red', 'orange', 'green', 'blue', 'yellow']

# Plot bar plot for F1 score per emotion per model
ax = f1_scores_per_emotions_df.plot(kind='bar', figsize=(14, 8), color=custom_colors)
plt.title("F1 Score per Emotion for Each Model")
plt.xlabel("Model")
plt.ylabel("F1 Score per Emotion")
plt.legend(title="Emotions", labels=emotion_labels)
plt.xticks(rotation=0)
plt.savefig(figure_dir / 'plot_14.png')
plt.close()


# In[52]:


# Calculate the correlation matrix
correlation_matrix = f1_scores_per_emotions_df.corr()

# Plot the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True, cbar_kws={'shrink': .8})
plt.title("Correlation Heatmap for Emotions F1 Scores")
plt.savefig(figure_dir / 'plot_15.png')
plt.close()


# In[127]:



# Load csv file 
new_data = pd.read_csv("C:/Users/Aggeliki/Desktop/Thesis/public_data_2/track_a/dev/eng.csv")

X_new = new_data['text']

# Load TF-IDF
tfidf_vectorizer = joblib.load("C:/Users/Aggeliki/Desktop/Thesis/public_data/tfidf_vectorizer.pkl")

X_new_tfidf = tfidf_vectorizer.transform(X_new)

# Load model
multi_label_model = joblib.load("C:/Users/Aggeliki/Desktop/Thesis/saved_model.pkl")

# Predictions 
predictions = multi_label_model.predict(X_new_tfidf)

columns_to_update = ['anger', 'fear', 'joy', 'sadness', 'surprise']  
if len(columns_to_update) != predictions.shape[1]:
    print("Error: The number of columns does not match the prediction shape.")
else:
    for i, col in enumerate(columns_to_update):
        new_data[col] = predictions[:, i]  

new_data.drop(columns=['text'], inplace=True)

# Save 
new_data.to_csv('C:/Users/Aggeliki/Desktop/Thesis/public_data/predictions_output.csv', index=False)

print("Save Predictions 'predictions_output.csv'.")


# In[128]:


pip install gensim


# In[129]:


#Use Word Embeddings using Word2Vec


# Suppress all warnings
warnings.filterwarnings('ignore')

# Sample data preprocessing
def preprocess_text(text):
    return text.lower().split()

# Convert text data into word embeddings using Word2Vec
def get_word2vec_embeddings(text_data, vector_size=100):
    tokenized_text = text_data.apply(preprocess_text)
    
    # Train Word2Vec on the tokenized text
    word2vec_model = Word2Vec(sentences=tokenized_text, vector_size=vector_size, window=5, min_count=1, workers=4)
    
    # Generate embeddings by averaging word vectors
    embeddings = []
    for text in tokenized_text:
        if len(text) > 0:
            vectors = [word2vec_model.wv[word] for word in text if word in word2vec_model.wv]
            if vectors:
                embeddings.append(np.mean(vectors, axis=0))
            else:
                embeddings.append(np.zeros(vector_size))
        else:
            embeddings.append(np.zeros(vector_size))
    return np.array(embeddings)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear', random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

# Convert text data to Word2Vec embeddings
X_train_embeddings = get_word2vec_embeddings(X_train)
X_test_embeddings = get_word2vec_embeddings(X_test)

results = []

print('Print only F1 per emotion:')

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
    
    #Samples-F1 
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

# Convert results to DataFrame for easy viewing and plotting
results_df = pd.DataFrame(results)
print('\nResult Table:')
print(results_df[['Model', 'Precision', 'Recall', 'F1 Score (Macro)', 'F1 Score (Micro)','Samples-F1', 'Accuracy']])

# Load existing Excel file
#excel_path = "C:\\Users\\Aggeliki\\Desktop\\Thesis\\results_summary.xlsx"

#print(f"Results saved to {excel_path}")


# In[130]:


get_ipython().system('pip install optuna')


# In[131]:


# Tune TF-IDF models using Optuna in LR 


# Suppress all warnings
warnings.filterwarnings('ignore')

# Initialize a list to store trial results
trial_results = []

# Define objective function for Optuna
def objective(trial):
    # Hyperparameters to tune
    max_features = trial.suggest_int('max_features', 1000, 5000, step=500)
    ngram_range = (trial.suggest_int('ngram_range_min', 1, 2), trial.suggest_int('ngram_range_max', 2, 3))

    # Initialize TfidfVectorizer with trial parameters
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Initialize Logistic Regression
    lr_model = LogisticRegression(max_iter=5000, class_weight='balanced', solver='lbfgs')

    # MultiOutputClassifier to handle multi-label classification
    multi_label_model = MultiOutputClassifier(lr_model)
    multi_label_model.fit(X_train_tfidf, Y_train)
    
    Y_pred = multi_label_model.predict(X_test_tfidf)
    
    # Calculate F1 score (Macro) for evaluation
    f1_macro = f1_score(Y_test, Y_pred, average='macro')

    # Save the trial's result
    trial_results.append({
        'max_features': max_features,
        'ngram_range': ngram_range,
        'f1_macro': f1_macro
    })
    
    return f1_macro

# Split data (assuming X and Y are already defined)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# Print best hyperparameters and best score
print("Best hyperparameters: ", study.best_params)
print("Best F1 score (Macro): ", study.best_value)

# Convert trial results to DataFrame
results_df = pd.DataFrame(trial_results)

# Save results to Excel
#excel_path = "C:\\Users\\Aggeliki\\Desktop\\Thesis\\optuna_tuning_results.xlsx"
#with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    #results_df.to_excel(writer, index=False, sheet_name="Tuning Results")

#print(f"Results saved to {excel_path}")


# In[132]:


#Tune SVM parameters with Optuna


# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize a list to store trial results
trial_results = []

def objective(trial):
    # Suggest hyperparameters
    C = trial.suggest_float("C", 1e-3, 10.0, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    degree = trial.suggest_int("degree", 2, 5) if kernel == "poly" else 3
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    
    # Initialize and train the model
    svm = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, random_state=42)
    multi_label_model = MultiOutputClassifier(svm)
    multi_label_model.fit(X_train_tfidf, Y_train)
    
    # Predict and calculate F1 score
    Y_pred = multi_label_model.predict(X_test_tfidf)
    f1_macro = f1_score(Y_test, Y_pred, average='macro')

    # Save trial results
    trial_results.append({
        "C": C,
        "kernel": kernel,
        "degree": degree if kernel == "poly" else None,
        "gamma": gamma,
        "f1_macro": f1_macro
    })
    
    return f1_macro

# Create Optuna study and optimize
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, n_jobs=-1)

# Best parameters and score
print("Best parameters:", study.best_params)
print("Best F1 Macro Score:", study.best_value)

# Train the best SVM with the optimal parameters
best_params = study.best_params
best_svm = SVC(
    C=best_params["C"],
    kernel=best_params["kernel"],
    degree=best_params.get("degree", 3),
    gamma=best_params["gamma"],
    random_state=42,
)
best_model = MultiOutputClassifier(best_svm)
best_model.fit(X_train_tfidf, Y_train)

# Evaluate the final model
Y_pred_best = best_model.predict(X_test_tfidf)
final_f1_macro = f1_score(Y_test, Y_pred_best, average='macro')
print("Final F1 Macro Score with best model:", final_f1_macro)

# Convert trial results to DataFrame
results_df = pd.DataFrame(trial_results)

# Save results and best parameters to Excel
#excel_path = "C:\\Users\\Aggeliki\\Desktop\\Thesis\\svm_tuning_results.xlsx"
#with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    #results_df.to_excel(writer, index=False, sheet_name="Tuning Results")
    
    # Save best parameters and final score in a separate sheet
    #best_params_df = pd.DataFrame([best_params])
    #best_params_df["final_f1_macro"] = final_f1_macro
    #best_params_df.to_excel(writer, index=False, sheet_name="Best Parameters")

#print(f"Results saved to {excel_path}")


# In[133]:


pip install transformers


# In[30]:

