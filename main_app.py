import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Sample data
data = pd.read_csv(r"C:\Users\Samson\Desktop\Pdf Loaders\updated_legal_data.csv")
    
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
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

accuracy_lr, precision_lr, recall_lr, f1_lr = evaluate_model(y_test, y_pred_lr)
accuracy_nb, precision_nb, recall_nb, f1_nb = evaluate_model(y_test, y_pred_nb)

print(f"Logistic Regression: Accuracy={accuracy_lr}, Precision={precision_lr}, Recall={recall_lr}, F1={f1_lr}")
print(f"Naive Bayes: Accuracy={accuracy_nb}, Precision={precision_nb}, Recall={recall_nb}, F1={f1_nb}")
