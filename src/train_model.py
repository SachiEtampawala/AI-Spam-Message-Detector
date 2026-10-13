import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Find project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "sms_spam.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "spam_model.pkl"

# Create the models folder if it does not exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_model():

    print("Loading dataset...")

    # Load the cleaned dataset
    df = pd.read_csv(DATA_PATH)

    X = df["message"]
    y = df["label"]

    print(f"Total messages: {len(df)}")
    print("\nPreparing training and testing data...")

    # Split data while preserving the spam/ham proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training messages: {len(X_train)}")
    print(f"Testing messages: {len(X_test)}")

    # Convert text into numerical features
    print("\nConverting text using TF-IDF...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train the machine learning model
    print("\nTraining Naive Bayes model...")

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Predict labels for the unseen test data
    y_pred = model.predict(X_test_tfidf)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, pos_label="spam", zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, pos_label="spam", zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, pos_label="spam", zero_division=0
    )

    print("\nMODEL EVALUATION")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nDetailed classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion matrix (labels: ham, spam):")
    print(confusion_matrix(y_test, y_pred, labels=["ham", "spam"]))

    # Save the trained model and fitted vectorizer together
    joblib.dump(
        {
            "model": model,
            "vectorizer": vectorizer
        },
        MODEL_PATH
    )

    print(f"\nModel saved successfully to: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()