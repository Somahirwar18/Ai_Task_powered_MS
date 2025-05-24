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
st.markdown("Enter a task or bug summary, and get predictions for multiple labels.")

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


        # # Display predictions
        # st.subheader("📌 Predicted Outputs:")
        # for label, pred in zip(output_labels, preds):
        #     st.write(f"**{label}:** {pred}")

        # # Pie charts
        # st.subheader("📊 Visualizations")
        # colors_list = ['skyblue', 'lightgreen', 'salmon', 'violet', 'orange', 'lightcoral']
        # for i, (label, pred) in enumerate(zip(output_labels, preds)):
        #     fig, ax = plt.subplots()
        #     color = colors_list[i % len(colors_list)]
        #     ax.pie([1], labels=[pred], autopct='%1.1f%%', colors=[color])
        #     st.markdown(f"**{label}**")
        #     st.pyplot(fig)

        # st.info("✅ Prediction complete. Scroll up to view the results.")

















# # # import streamlit as st
# # # import joblib
# # # import re
# # # import nltk
# # # from nltk.tokenize import word_tokenize
# # # from nltk.corpus import stopwords
# # # from nltk.stem import PorterStemmer

# # # nltk.download('punkt')
# # # nltk.download('stopwords')

# # # # Load assets
# # # model = joblib.load('model.pkl')
# # # vectorizer = joblib.load('tfidf_vectorizer.pkl')
# # # output_labels = joblib.load('multi_output_labels.pkl')

# # # stemmer = PorterStemmer()
# # # stop_words = set(stopwords.words('english'))

# # # def preprocess(text):
# # #     text = text.lower()
# # #     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
# # #     tokens = word_tokenize(text)
# # #     return ' '.join(stemmer.stem(word) for word in tokens if word not in stop_words)

# # # st.set_page_config(page_title="Multi-Label Issue Predictor", layout="centered")
# # # st.title("🧠 Multi-Label Issue Predictor")
# # # st.markdown("Enter a bug **summary** and get predictions for multiple labels.")

# # # summary_input = st.text_area("✍️ Enter Summary:")

# # # if st.button("Predict"):
# # #     if not summary_input.strip():
# # #         st.warning("Please enter a valid summary.")
# # #     else:
# # #         clean_input = preprocess(summary_input)
# # #         vect_input = vectorizer.transform([clean_input])
# # #         prediction = model.predict(vect_input)[0]
# # #         result = dict(zip(output_labels, prediction))
# # #         st.success("✅ Prediction Results:")
# # #         for key, value in result.items():
# # #             st.markdown(f"**{key}:** {value}")
# # #         st.markdown("### Note:")
# # import streamlit as st
# # import joblib
# # import re
# # import nltk
# # from nltk.tokenize import word_tokenize
# # from nltk.corpus import stopwords
# # from nltk.stem.porter import PorterStemmer
# # import matplotlib.pyplot as plt

# # # Download nltk data once
# # nltk.download('punkt')
# # nltk.download('stopwords')

# # # Load model, vectorizer, and output labels list
# # model = joblib.load('model.pkl')
# # vectorizer = joblib.load('tfidf_vectorizer.pkl')
# # output_labels = joblib.load('multi_output_labels.pkl')  # e.g. ['Issue Type', 'Status', 'Project Name', 'Project Type', 'Priority', 'Resolution']

# # stop_words = set(stopwords.words('english'))
# # stemmer = PorterStemmer()

# # def preprocess(text):
# #     text = text.lower()
# #     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
# #     tokens = word_tokenize(text)
# #     filtered = [stemmer.stem(t) for t in tokens if t not in stop_words]
# #     return ' '.join(filtered)

# # st.title("🔮 Multi-Output Task Issue Predictor")

# # summary_input = st.text_area("Enter Task Summary:")

# # if st.button("Predict"):
# #     if not summary_input.strip():
# #         st.warning("Please enter task summary.")
# #     else:
# #         processed = preprocess(summary_input)
# #         vect = vectorizer.transform([processed])
# #         preds = model.predict(vect)[0]
        
# #         # Show results nicely
# #         st.subheader("Predicted Outputs:")
# #         for label, pred in zip(output_labels, preds):
# #             st.write(f"**{label}:** {pred}")
        
# #         # Simple visualization: Pie chart of Priority (or any categorical output)
# #         if 'Priority' in output_labels:
# #             pred_priority = preds[output_labels.index('Priority')]
# #             st.subheader("Priority Visualization")
# #             fig, ax = plt.subplots()
# #             ax.pie([1], labels=[pred_priority], autopct='%1.1f%%', colors=['skyblue'])
# #             st.pyplot(fig)
# #         st.markdown("### Note:")


# # _____________________
# import  numpy as np
# import pandas as pd
# import streamlit as st
# import joblib
# import pandas as pd
# import re
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from sklearn.metrics import accuracy_score, hamming_loss
# import matplotlib.pyplot as plt

# # Download NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')

# # Load model components
# model = joblib.load('model.pkl')
# vectorizer = joblib.load('tfidf_vectorizer.pkl')
# output_labels = joblib.load('multi_output_labels.pkl')  # List of target labels
# # # Load training data (for evaluation)
# # df = pd.read_csv('JIRA_FINAL_cleaned.csv')
# # df.dropna(subset=['clean_summary'], inplace=True)
# # Y = df[['clean_issue_type', 'clean_status', 'clean_project_name', 'clean_project_type', 'clean_priority', 'clean_resolution']]
# # X_vect = vectorizer.transform(df['clean_summary'])  # Assuming Summary is already cleaned

# # Preprocessing function
# stop_words = set(stopwords.words('english'))
# stemmer = PorterStemmer()
# def preprocess(text):
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
#     tokens = word_tokenize(text)
#     filtered = [stemmer.stem(t) for t in tokens if t not in stop_words]
#     return ' '.join(filtered)

# # UI
# st.set_page_config(page_title="Multi-Label Issue Predictor", layout="centered")
# # st.title("🔮 Multi-Output Task Issue Predictor")
# st.title(" 🔮 AI-Powered Task Management System")
# st.markdown("This app predicts multiple task labels based on a bug/task summary.")

# # Input field
# summary_input = st.text_area("✍️ Enter Task Summary:")

# if st.button("🎯 Predict"):
#     if not summary_input.strip():
#         st.warning("Please enter a valid task summary.")
#     else:
#         processed = preprocess(summary_input)
#         vect = vectorizer.transform([processed])
#         preds = model.predict(vect)[0]

#         # Display predictions
#         st.subheader("📌 Predicted Outputs:")
#         for label, pred in zip(output_labels, preds):
#             st.write(f"**{label}:** {pred}")

#         # Optional pie chart for Priority
#         if 'clean_priority' in output_labels:
#             pred_priority = preds[output_labels.index('clean_priority')]
#             st.subheader("📊 Priority Pie Chart")
#             fig, ax = plt.subplots()
#             ax.pie([1], labels=[pred_priority], autopct='%1.1f%%', colors=['skyblue'])
#             st.pyplot(fig)
#             # Define a list of distinct colors to cycle through
#             colors_list = ['skyblue', 'lightgreen', 'salmon', 'violet', 'orange', 'lightcoral']
#             st.subheader("Predicted Outputs:")
#             for i, (label, pred) in enumerate(zip(output_labels, preds)):
#                 st.write(f"**{label}:** {pred}")
    
#                 st.subheader(f"📊 {label} Pie Chart")
#                 fig, ax = plt.subplots()
    
#                 # Use color cycling based on index, modulo number of colors
#                 color = colors_list[i % len(colors_list)]
    
#                 ax.pie([1], labels=[pred], autopct='%1.1f%%', colors=[color])
#                 st.pyplot(fig)
#             # st.subheader("Predicted Outputs:")
#             # for label, pred in zip(output_labels, preds):
#             #     st.write(f"**{label}:** {pred}")
#             #     st.subheader(f"📊 {label} Pie Chart")
#             #     fig, ax = plt.subplots()
#             #     ax.pie([1], labels=[pred], autopct='%1.1f%%', colors=['skyblue'])
#             #     st.pyplot(fig)

# # Evaluation block
# # if st.button("📈 Show Evaluation Metrics"):
#     # y_pred = model.predict(X_vect)
#     # y_pred_df = pd.DataFrame(y_pred, columns=Y.columns)
#     # accuracies = {col: accuracy_score(Y[col], y_pred_df[col]) for col in Y.columns}
#     # hamming = hamming_loss(Y, y_pred)

#     # st.markdown("### ✅ Evaluation Results")
#     # st.write(f"**Exact Match Accuracy**: `{accuracy_score(Y, y_pred):.4f}`")
#     # st.write(f"**Hamming Loss**: `{hamming:.4f}`")

#     # Bar chart for individual label accuracies
#     # fig, ax = plt.subplots()
#     # ax.bar(accuracies.keys(), accuracies.values(), color='mediumseagreen')
#     # ax.set_ylabel('Accuracy')
#     # ax.set_ylim(0, 1)
#     # ax.set_title('Per-Label Accuracy')
#     # plt.xticks(rotation=45)
#     # st.pyplot(fig)
