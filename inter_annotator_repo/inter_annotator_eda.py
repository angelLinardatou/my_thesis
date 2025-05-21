import glob
import math
import matplotlib.pyplot as plt 
import numpy as np 
import os
import pandas as pd
import seaborn as sns 

from pathlib import Path
import os

# Define base paths
base_dir = Path('C:/Users/Aggeliki/Desktop/Thesis')
annotation_dir = base_dir / 'annotations'
figure_dir = base_dir / 'eda_figures'
figure_dir.mkdir(exist_ok=True)

# Replace glob and os usage
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

xlsx_files = list(Path(annotation_dir).glob('*.xlsx'))
dataframes = {f.name: pd.read_excel(f, skiprows=1) for f in xlsx_files}

# Rename columns for consistency
for fname, df in dataframes.items():
    df.columns = ['id', 'text', 'anger', 'fear', 'joy', 'sadness', 'surprise']

#EDA for the annotations in dev.csv data 


#Load Data 
folder_path = 'C:/Users/Aggeliki/Desktop/Thesis/annotations'  


# In[2]:


csv_files =  glob.glob(os.path.join(folder_path, '*.xlsx'))


# In[3]:


dataframes = {}

for file in csv_files:
    file_name = os.path.basename(file)  
    dataframes[file_name] = pd.read_excel(file , skiprows=1)
    
# Confirm loaded files and display basic info for each
print("Loaded files:", list(dataframes.keys()))


# Rename the columns to meaningful names for each dataframe
for file_name, df in dataframes.items():
    df.columns = ['id', 'text', 'anger', 'fear', 'joy', 'sadness', 'surprise']  


# In[4]:


# Look for missing values 
missing_values_summary = {file_name: df.isnull().sum().sum() for file_name, df in dataframes.items()}

# Dictionary to store rows with missing values for each file before correction
missing_rows_before_correction = {}

for file_name, df in dataframes.items():
    # Select rows with any missing value
    rows_with_missing_values = df[df.isnull().any(axis=1)]
    if not rows_with_missing_values.empty:
        missing_rows_before_correction[file_name] = rows_with_missing_values

# Fix missing values with mean
for file_name, df in dataframes.items():
    for column in ['anger', 'fear', 'joy', 'sadness', 'surprise']:
        if column in df.columns:
            mean_value = df[column].mean()  
            df[column].fillna(mean_value, inplace=True)

# Display the rows with missing values before correction
if missing_rows_before_correction:
    for file_name, rows in missing_rows_before_correction.items():
        print(f"Rows with missing values in file: {file_name}")
        print(rows)
        print("-" * 50)
else:
    print("No missing values found in any file.")


# In[5]:


# Basic statistics for emotion columns
   
for file_name, df in dataframes.items():
   print('\nFile Name:', file_name)
   print("\nBasic Statistics (Emotions):")
   print(df[['anger', 'fear', 'joy', 'sadness', 'surprise']].describe())


# In[6]:


# How many times each emotion appears (1) in the data set, for example anger how many times appears in all the texts 

for index, (file_name, df) in enumerate(dataframes.items(), start=1):
    annotator_label = f'Annotator {index}'
    print('Annotator:', annotator_label)
    print("\nEmotion Distribution:")
    
    emotion_counts = df[['anger', 'fear', 'joy', 'sadness', 'surprise']].sum()
    print(emotion_counts)

    plt.figure(figsize=(10, 6)) 
    emotion_counts.plot(kind='bar', color=['red', 'blue', 'green', 'yellow', 'purple'])

    plt.title('Emotion Distribution')  
    plt.xlabel('Emotion')  
    plt.ylabel('Count')  
    plt.xticks(rotation=0)  

plt.savefig(figure_dir / 'plot_01.png')
plt.close()


# In[7]:


#Total emotions appears in all the annotators 
total_emotion_counts = sum(df[['anger', 'fear', 'joy', 'sadness', 'surprise']].sum() for df in dataframes.values())

plt.figure(figsize=(10, 6))
total_emotion_counts.plot(kind='bar', color=['red', 'blue', 'green', 'yellow', 'purple'])

plt.title('Total Emotion Distribution Across All Annotators')
plt.xlabel('Emotion')
plt.ylabel('Total Count')
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(figure_dir / 'plot_02.png')
plt.close()


# In[8]:


#For each emotions how many times appears in all the annotators 

emotions = ['anger', 'fear', 'joy', 'sadness', 'surprise']


emotion_counts_per_file = {
    emotion: [df[emotion].sum() for df in dataframes.values()] for emotion in emotions
}

# Bar plot 
for emotion, counts in emotion_counts_per_file.items():
    plt.figure(figsize=(8, 5))
    annotator_labels = [f'Annotator {i+1}' for i in range(len(dataframes))]
    plt.bar(annotator_labels, counts, color='skyblue')
    plt.title(f'{emotion.capitalize()} Distribution Across Annotators')
    plt.xlabel('Annotators') 
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()

plt.savefig(figure_dir / 'plot_03.png')
plt.close()


# In[9]:


#Corellation between annotators 
emotion_data = {
    file_name: df[['anger', 'fear', 'joy', 'sadness', 'surprise']].sum()
    for file_name, df in dataframes.items()
}
emotion_df = pd.DataFrame(emotion_data).T  
emotion_df.index.name = "Files"

correlation_matrix = emotion_df.corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Heatmap 
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title("Upper Triangular Heatmap of Correlation Between Annotators Based on Emotion Distribution")
plt.tight_layout()
plt.savefig(figure_dir / 'plot_04.png')
plt.close()


# In[10]:


#Inter-annotator agreement with Cohen's k 