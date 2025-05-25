import streamlit as st
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt

# Download NLTK data if not already present
# Download NLTK data if not already present
import nltk
nltk.download('averaged_perceptron_tagger')
# Or to download all packages + data + docs:

nltk.download('all')

nltk.download('punkt')
nltk.download('stopwords')

# Load saved model and vectorizer
model = joblib.load('model.pkl')  # Trained MultiOutputClassifier
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # SAME vectorizer used during training
output_labels = joblib.load('multi_output_labels.pkl')  # List of label names

# Preprocessing setup
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    filtered = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered)

# Streamlit UI
st.set_page_config(page_title="AI-Powered Task Management System", layout="centered")
st.title("🔮 AI-Powered Task Management System")
st.markdown("Enter a task or bug summary Description, and get predictions for multiple labels.")

summary_input = st.text_area("✍️ Enter Task Summary:")

if st.button("🎯 Predict"):
    if not summary_input.strip():
        st.warning("Please enter a valid task summary.")
    else:
        # Preprocess and transform input
        processed = preprocess(summary_input)
        vect = vectorizer.transform([processed])

        # Optional debug: check feature shape
        # st.write(f"Input shape: {vect.shape}, Model expects: {model.estimators_[0].coef_.shape[1]}")

        # Predict
        preds = model.predict(vect)[0]
        
        # Display predictions
        st.subheader("📌 Predicted Outputs:")
        for label, pred in zip(output_labels, preds):
            st.write(f"**{label}:** {pred}")

# Combined Pie Chart
        st.subheader("📊 Combined Prediction Summary")

# Combine label and prediction for pie chart labels
        combined_labels = [f"{label}: {pred}" for label, pred in zip(output_labels, preds)]

# Each prediction gets equal share (100% divided by number of predictions)
        percentages = [100 / len(combined_labels)] * len(combined_labels)

# Define colors
        colors_list = ['skyblue', 'lightgreen', 'salmon', 'violet', 'orange', 'lightcoral']

# Create single pie chart
        fig, ax = plt.subplots()
        ax.pie(percentages, labels=combined_labels, autopct='%1.1f%%', colors=colors_list[:len(combined_labels)])
        ax.axis('equal')  # Ensure pie is a circle

        st.pyplot(fig)
        st.info("✅ Combined prediction chart displayed successfully.")

