import joblib
from pathlib import Path

# Locate the trained model
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "spam_model.pkl"

def load_model():
    """Load the trained model and text vectorizer."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. "
            "Please run src/train_model.py first."
        )

    saved_objects = joblib.load(MODEL_PATH)

    return saved_objects["model"], saved_objects["vectorizer"]

def predict_message(message, model, vectorizer):
    """Classify a message as spam or ham."""

    message = message.strip()

    if not message:
        raise ValueError("Please enter a message.")

    message_features = vectorizer.transform([message])

    prediction = model.predict(message_features)[0]

    probabilities = model.predict_proba(message_features)[0]

    probability_by_label = dict(
        zip(model.classes_, probabilities)
    )

    return {
        "prediction": prediction,
        "spam_probability": float(
            probability_by_label.get("spam", 0.0)
        ),
        "ham_probability": float(
            probability_by_label.get("ham", 0.0)
        )
    }