import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    make_scorer
)

# 1. Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "sms_spam.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "spam_model.pkl"

# 2. Load and validate the dataset
def load_dataset():

    print("=" * 60)
    print("AI SPAM MESSAGE DETECTOR")
    print("=" * 60)

    print("\nLoading dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run this command first:\n"
            "python src/prepare_data.py"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {"label", "message"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Dataset must contain 'label' and 'message' columns."
        )

    df = df[["label", "message"]].copy()

    df = df.dropna(subset=["label", "message"])

    df["label"] = df["label"].astype(str).str.strip()
    df["message"] = df["message"].astype(str).str.strip()

    df = df[df["message"] != ""]

    df = df[df["label"].isin(["ham", "spam"])]

    df = df.drop_duplicates(subset=["label", "message"])

    label_counts = df["label"].value_counts()

    if not {"ham", "spam"}.issubset(label_counts.index):
        raise ValueError(
            "The dataset must contain both ham and spam messages."
        )

    if label_counts.min() < 5:
        raise ValueError(
            "Each class must have at least five messages "
            "for five-fold cross-validation."
        )

    print("\nDataset loaded successfully.")
    print(f"Total messages: {len(df)}")

    print("\nMessage distribution:")
    print(df["label"].value_counts())

    return df

# 3. Train and tune the model
def train_model():

    df = load_dataset()

    X = df["message"]
    y = df["label"]

    print("\nSplitting the dataset...")

    # Keep the test set separate until final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training messages: {len(X_train)}")
    print(f"Testing messages: {len(X_test)}")

    # 4. Create a machine learning pipeline
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=10000,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            MultinomialNB()
        )
    ])

    # Compare different smoothing strengths.
    parameter_grid = {
        "classifier__alpha": [
            0.01,
            0.05,
            0.1,
            0.2,
            0.5,
            1.0,
            2.0
        ]
    }

    # F1-score for the spam class.
    spam_f1_scorer = make_scorer(
        f1_score,
        pos_label="spam",
        zero_division=0
    )

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # 5. Search for the best hyperparameter
    print("\nStarting five-fold cross-validation...")
    print("Comparing seven alpha values.")
    print("This may take a little time.")

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring=spam_f1_scorer,
        cv=cross_validation,
        n_jobs=-1,
        refit=True,
        return_train_score=False
    )

    search.fit(X_train, y_train)

    print("\nCross-validation completed.")

    print("\nBest parameters:")
    print(search.best_params_)

    print(
        "Best cross-validation spam F1-score: "
        f"{search.best_score_:.4f}"
    )

    # Show the scores for all tested alpha values.
    results = pd.DataFrame(search.cv_results_)

    comparison = results[
        [
            "param_classifier__alpha",
            "mean_test_score",
            "std_test_score"
        ]
    ].copy()

    comparison.columns = [
        "Alpha",
        "Mean Spam F1-score",
        "Standard Deviation"
    ]

    comparison = comparison.sort_values(
        by="Mean Spam F1-score",
        ascending=False
    )

    print("\nHyperparameter comparison:")
    print(comparison.to_string(index=False))

    # The best estimator has already been refitted
    # on the complete training dataset.
    best_model = search.best_estimator_

    # 6. Evaluate once on the held-out test dataset
    print("\nEvaluating the selected model on test data...")

    y_pred = best_model.predict(X_test)

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

    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)

    print(f"Accuracy:  {accuracy:.4f} ({accuracy:.2%})")
    print(f"Precision: {precision:.4f} ({precision:.2%})")
    print(f"Recall:    {recall:.4f} ({recall:.2%})")
    print(f"F1-score:  {f1:.4f} ({f1:.2%})")

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=["ham", "spam"],
            target_names=["Legitimate (ham)", "Spam"],
            zero_division=0
        )
    )

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

    print("Confusion matrix:")
    print(confusion_df)

    # 7. Save the complete pipeline
    print("\nSaving the trained pipeline...")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        best_model,
        MODEL_PATH
    )

    print("\nModel saved successfully.")
    print(f"Location: {MODEL_PATH}")

    print("\nTraining and evaluation completed!")

# 8. Run the training process
if __name__ == "__main__":
    train_model()