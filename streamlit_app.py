import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from rank_bm25 import BM25Okapi

# Function to evaluate model
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Streamlit App
st.title('AI Models for Legal Text Classification and Query Matching')

# File uploader
uploaded_file = st.file_uploader(r"C:\Users\Samson\Desktop\Pdf Loaders\updated_legal_data.csv", type=["csv"])

if uploaded_file is not None:
    # Read the data
    data = pd.read_csv(uploaded_file)
    df = pd.DataFrame(data)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Raw Content'])
    y = df['Offense']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    # Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)

    # Evaluation
    accuracy_lr, precision_lr, recall_lr, f1_lr = evaluate_model(y_test, y_pred_lr)
    accuracy_nb, precision_nb, recall_nb, f1_nb = evaluate_model(y_test, y_pred_nb)

    # BM25 Vectorization
    documents = df['Raw Content'].tolist()
    bm25 = BM25Okapi([doc.split() for doc in documents])

    # Display evaluation results
    st.header('Text Classification')
    st.subheader('Logistic Regression')
    st.write(f"Accuracy: {accuracy_lr}")
    st.write(f"Precision: {precision_lr}")
    st.write(f"Recall: {recall_lr}")
    st.write(f"F1 Score: {f1_lr}")

    st.subheader('Naive Bayes')
    st.write(f"Accuracy: {accuracy_nb}")
    st.write(f"Precision: {precision_nb}")
    st.write(f"Recall: {recall_nb}")
    st.write(f"F1 Score: {f1_nb}")

    # Query Matching
    st.header('Query Matching')
    query = st.text_input('Enter your query:')
    if query:
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)
        best_match = documents[scores.argmax()]
        st.write(f"Best Match: {best_match}")

