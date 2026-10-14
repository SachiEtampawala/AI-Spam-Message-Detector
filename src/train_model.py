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

# 1. Set project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "sms_spam.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "spam_model.pkl"

# 2. Load and validate the dataset
def load_dataset():

    print("=" * 50)
    print("AI SPAM MESSAGE DETECTOR")
    print("=" * 50)

    print("\nLoading dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed dataset not found.\n"
            "Please run this command first:\n"
            "python src/prepare_data.py"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {"label", "message"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Dataset must contain 'label' and 'message' columns."
        )

    # Keep only the required columns.
    df = df[["label", "message"]].copy()

    # Remove missing values.
    df = df.dropna(subset=["label", "message"])

    # Convert values to strings and remove extra spaces.
    df["label"] = df["label"].astype(str).str.strip()
    df["message"] = df["message"].astype(str).str.strip()

    # Remove empty messages.
    df = df[df["message"] != ""]

    # Keep only the supported labels.
    df = df[df["label"].isin(["ham", "spam"])]

    # Remove duplicate messages.
    df = df.drop_duplicates(subset=["label", "message"])

    # Ensure both classes have enough examples for splitting.
    label_counts = df["label"].value_counts()

    if not {"ham", "spam"}.issubset(label_counts.index):
        raise ValueError(
            "The dataset must contain both ham and spam messages."
        )

    if label_counts.min() < 2:
        raise ValueError(
            "Each class must contain at least two messages."
        )

    print("\nDataset loaded successfully!")
    print(f"Total messages: {len(df)}")

    print("\nMessage distribution:")
    print(df["label"].value_counts())

    return df

# 3. Train and evaluate the machine learning model

def train_model():

    df = load_dataset()

    X = df["message"]
    y = df["label"]

    print("\nSplitting the dataset...")

    # Keep 80% for training and 20% for testing.
    # Stratification preserves the approximate class proportions.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training messages: {len(X_train)}")
    print(f"Testing messages: {len(X_test)}")

    # 4. Convert messages into numerical features
    print("\nConverting messages into TF-IDF features...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True
    )

    # Learn the vocabulary from the training data only.
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Transform the test data using the same vocabulary.
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"Number of features: {len(vectorizer.vocabulary_)}")

    # 5. Train the Naive Bayes classifier
    print("\nTraining the Multinomial Naive Bayes model...")

    model = MultinomialNB(alpha=0.5)

    model.fit(X_train_tfidf, y_train)

    print("Model training completed!")

    # 6. Evaluate the model using unseen test messages

    print("\nGenerating predictions...")

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        pos_label="spam",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label="spam",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="spam",
        zero_division=0
    )

    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)

    print(f"Accuracy:  {accuracy:.4f} ({accuracy:.2%})")
    print(f"Precision: {precision:.4f} ({precision:.2%})")
    print(f"Recall:    {recall:.4f} ({recall:.2%})")
    print(f"F1-score:  {f1:.4f} ({f1:.2%})")

    print("\nDetailed Classification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=["ham", "spam"],
            target_names=["Legitimate (ham)", "Spam"],
            zero_division=0
        )
    )

    # Rows represent actual labels.
    # Columns represent predicted labels.
    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=["ham", "spam"]
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=["Actual Ham", "Actual Spam"],
        columns=["Predicted Ham", "Predicted Spam"]
    )

    print("Confusion Matrix:")
    print(confusion_df)

    # 7. Save the trained model and vectorizer
    print("\nSaving the trained model...")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "vectorizer": vectorizer
        },
        MODEL_PATH
    )

    print(f"\nModel saved successfully!")
    print(f"Model location: {MODEL_PATH}")

    print("\nTraining and evaluation completed successfully!")

# 8. Run the training pipeline
if __name__ == "__main__":
    train_model()