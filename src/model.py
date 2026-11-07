from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_dataset(path):
    """Read and validate the labeled message CSV."""
    data = pd.read_csv(path)
    required = {"text", "label"}
    if not required.issubset(data.columns):
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
    data = data.dropna(subset=["text", "label"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.strip().str.title()
    data = data[data["label"].isin(["Ham", "Spam"])]
    data = data[data["text"] != ""]
    if data["label"].nunique() != 2 or data["label"].value_counts().min() < 2:
        raise ValueError("Dataset needs at least two examples of both Ham and Spam.")
    return data

def train_model(data):
    """Train TF-IDF + Multinomial Naive Bayes and retain a test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.25, random_state=42, stratify=data["label"]
    )
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
    model = MultinomialNB()
    X_train_vec = vectorizer.fit_transform(X_train)
    model.fit(X_train_vec, y_train)
    return model, vectorizer, X_test, y_test

def predict_message(message, model, vectorizer):
    """Return predicted class and the probability assigned to that class."""
    features = vectorizer.transform([message])
    label = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    class_index = list(model.classes_).index(label)
    return label, float(probabilities[class_index])

def evaluate_model(model, vectorizer, X_test, y_test):
    X_test_vec = vectorizer.transform(X_test)
    predictions = model.predict(X_test_vec)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label="Spam", zero_division=0),
        "recall": recall_score(y_test, predictions, pos_label="Spam", zero_division=0),
        "f1_score": f1_score(y_test, predictions, pos_label="Spam", zero_division=0),
    }
    matrix = confusion_matrix(y_test, predictions, labels=["Ham", "Spam"])
    return metrics, matrix
