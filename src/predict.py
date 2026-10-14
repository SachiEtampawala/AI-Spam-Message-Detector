import joblib

from pathlib import Path

# 1. Locate the saved model
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

# 3. Predict a message
def predict_message(message):

    if not isinstance(message, str):
        raise TypeError("Message must be a string.")

    message = message.strip()

    if not message:
        raise ValueError("Please enter a message to analyze.")

    model = load_model()

    # Predict the message category.
    prediction = model.predict([message])[0]

    # Get probabilities for both classes.
    probabilities = model.predict_proba([message])[0]

    # Find the probability belonging to each class.
    classes = list(model.classes_)

    spam_probability = float(
        probabilities[classes.index("spam")]
    )

    ham_probability = float(
        probabilities[classes.index("ham")]
    )

    return {
        "prediction": prediction,
        "spam_probability": spam_probability,
        "ham_probability": ham_probability
    }