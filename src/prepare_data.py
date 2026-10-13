import pandas as pd
from pathlib import Path

# Find the project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Define dataset paths
RAW_DATA_PATH = PROJECT_DIR / "data" / "SMSSpamCollection"
PROCESSED_DATA_PATH = PROJECT_DIR / "data" / "sms_spam.csv"

def prepare_dataset():
    print("Loading the SMS Spam Collection dataset...")

    # Load the original dataset
    df = pd.read_csv(
        RAW_DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )

    print("\nOriginal dataset loaded successfully!")

    # Remove missing and duplicate messages
    df = df.dropna()
    df = df.drop_duplicates()

    # Keep only valid labels
    df = df[df["label"].isin(["ham", "spam"])]

    # Save the prepared dataset
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("\nDataset preparation completed!")
    print(f"Total messages: {len(df)}")
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
    print(f"\nSaved dataset to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    prepare_dataset()