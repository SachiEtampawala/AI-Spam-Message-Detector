import joblib

from pathlib import Path

# 1. Locate the trained model
PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "spam_model.pkl"

# 2. Load the trained pipeline
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found.\n"
            "Please run: python src/train_model.py"
        )

    return joblib.load(MODEL_PATH)

# 3. Predict a message using an adjustable threshold
def predict_message(message, model, threshold=0.50):

    if not isinstance(message, str):
        raise TypeError(
            "Message must be a string."
        )

    message = message.strip()

    if not message:
        raise ValueError(
            "Please enter a message."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    # Get the model's probability estimates.
    probabilities = model.predict_proba([message])[0]

    classes = list(model.classes_)

    if "spam" not in classes or "ham" not in classes:
        raise ValueError(
            "The model must support both spam and ham classes."
        )

    spam_probability = float(
        probabilities[classes.index("spam")]
    )

    ham_probability = float(
        probabilities[classes.index("ham")]
    )

    # Apply the user-selected threshold.
    if spam_probability >= threshold:
        prediction = "spam"
    else:
        prediction = "ham"

    return {
        "prediction": prediction,
        "spam_probability": spam_probability,
        "ham_probability": ham_probability,
        "threshold": threshold
    }