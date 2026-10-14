import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

from predict import load_model

# 1. Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "threshold_test_cases.csv"

# 2. Load evaluation messages
def load_test_cases():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Evaluation dataset not found.\n"
            "Please create data/threshold_test_cases.csv first."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {"label", "message"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "CSV must contain 'label' and 'message' columns."
        )

    df = df.dropna(subset=["label", "message"]).copy()

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()

    df = df[
        df["label"].isin(["ham", "spam"])
        & (df["message"] != "")
    ]

    if df.empty:
        raise ValueError("The evaluation dataset is empty.")

    if set(df["label"]) != {"ham", "spam"}:
        raise ValueError(
            "The evaluation dataset must contain both ham and spam."
        )

    return df

# 3. Evaluate different thresholds
def evaluate_thresholds():

    print("=" * 65)
    print("SPAM DETECTION THRESHOLD EVALUATION")
    print("=" * 65)

    df = load_test_cases()

    model = load_model()

    messages = df["message"].tolist()
    actual_labels = df["label"].tolist()

    print(f"\nEvaluation messages: {len(df)}")
    print(f"Legitimate messages: {(df['label'] == 'ham').sum()}")
    print(f"Phishing-style messages: {(df['label'] == 'spam').sum()}")

    # Get probabilities once, then reuse them at every threshold.
    probabilities = model.predict_proba(messages)

    classes = list(model.classes_)

    if "spam" not in classes:
        raise ValueError(
            "The trained model does not contain the spam class."
        )

    spam_index = classes.index("spam")

    spam_probabilities = probabilities[:, spam_index]

    # A higher threshold makes the classifier more conservative
    # about labelling a message as spam.
    thresholds = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90
    ]

    results = []

    print("\nEvaluating thresholds...")

    for threshold in thresholds:

        predictions = [
            "spam" if probability >= threshold else "ham"
            for probability in spam_probabilities
        ]

        actual_spam = [
            label == "spam"
            for label in actual_labels
        ]

        predicted_spam = [
            label == "spam"
            for label in predictions
        ]

        true_positives = sum(
            actual and predicted
            for actual, predicted in zip(
                actual_spam,
                predicted_spam
            )
        )

        false_positives = sum(
            not actual and predicted
            for actual, predicted in zip(
                actual_spam,
                predicted_spam
            )
        )

        true_negatives = sum(
            not actual and not predicted
            for actual, predicted in zip(
                actual_spam,
                predicted_spam
            )
        )

        false_negatives = sum(
            actual and not predicted
            for actual, predicted in zip(
                actual_spam,
                predicted_spam
            )
        )

        precision = precision_score(
            actual_labels,
            predictions,
            pos_label="spam",
            zero_division=0
        )

        recall = recall_score(
            actual_labels,
            predictions,
            pos_label="spam",
            zero_division=0
        )

        f1 = f1_score(
            actual_labels,
            predictions,
            pos_label="spam",
            zero_division=0
        )

        results.append({
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "False Positives": false_positives,
            "Missed Spam": false_negatives,
            "True Positives": true_positives,
            "True Negatives": true_negatives
        })

    results_df = pd.DataFrame(results)

    print("\nTHRESHOLD COMPARISON")
    print("-" * 100)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Threshold": "{:.2f}".format,
                "Precision": "{:.2%}".format,
                "Recall": "{:.2%}".format,
                "F1-score": "{:.2%}".format
            }
        )
    )

    # Save the results for later analysis.
    output_path = (
        PROJECT_DIR / "outputs" / "threshold_results.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(f"\nResults saved to: {output_path}")

    # Show how the selected threshold classifies each message.
    print("\nMESSAGE-LEVEL RESULTS")
    print("-" * 65)

    message_results = pd.DataFrame({
        "Actual Label": actual_labels,
        "Spam Probability": spam_probabilities
    })

    message_results["Message"] = messages

    # Use 0.50 as a reference threshold for this detailed view.
    reference_threshold = 0.50

    message_results["Predicted Label"] = [
        "spam" if probability >= reference_threshold else "ham"
        for probability in spam_probabilities
    ]

    message_results["Result"] = [
        "Correct" if actual == predicted else "Incorrect"
        for actual, predicted in zip(
            message_results["Actual Label"],
            message_results["Predicted Label"]
        )
    ]

    print(
        message_results[
            [
                "Actual Label",
                "Predicted Label",
                "Spam Probability",
                "Result",
                "Message"
            ]
        ].to_string(
            index=False,
            formatters={
                "Spam Probability": "{:.2%}".format
            }
        )
    )

    print("\nEvaluation completed.")

# 4. Run the evaluation
if __name__ == "__main__":
    evaluate_thresholds()