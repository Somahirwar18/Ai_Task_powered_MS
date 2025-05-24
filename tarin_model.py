import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
import joblib

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load your cleaned CSV
df = pd.read_csv('jira_dataset.csv')  # Adjust filename as needed

# Drop rows with missing summaries or labels
df.dropna(subset=['clean_summary'], inplace=True)

# Labels to predict
required_columns = ['issue_type', 'status','project_name', 'project_type','project_lead','project_description', 'priority','resolution','task_assignee','task_deadline']
# label_cols = ['clean_issue_type', 'clean_status', 'clean_project_name', 
            #   'clean_project_type', 'clean_priority', 'clean_resolution']
df.dropna(subset=required_columns, inplace=True)

label_cols = required_columns[1:]
# Preprocess function
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    filtered = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered)

# Apply preprocessing
df['processed_summary'] = df['clean_summary'].apply(preprocess)

# Split features and labels
X = df['processed_summary']
Y = df[label_cols]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_vect = vectorizer.fit_transform(X)

# Train/Test split (optional)
X_train, X_test, Y_train, Y_test = train_test_split(X_vect, Y, test_size=0.2, random_state=42)

# Multi-output classification with Logistic Regression
base_model = LogisticRegression(max_iter=1000)
multi_model = MultiOutputClassifier(base_model)
multi_model.fit(X_train, Y_train)

# Save model and vectorizer
joblib.dump(multi_model, 'model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(label_cols, 'multi_output_labels.pkl')

print("✅ Model, vectorizer, and label list saved successfully!")
