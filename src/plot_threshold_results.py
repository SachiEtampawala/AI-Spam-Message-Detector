from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_DIR / "outputs" / "threshold_results.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Check whether evaluation results exist
if not RESULTS_PATH.exists():
    raise FileNotFoundError(
        "Threshold results not found. "
        "Run: python src/evaluate_thresholds.py"
    )

# Load evaluation results
df = pd.read_csv(RESULTS_PATH)

# Clean column names
df.columns = df.columns.str.strip()

# Check required columns
required_columns = [
    "Threshold",
    "Precision",
    "Recall",
    "F1-score",
    "False Positives",
    "Missed Spam",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in threshold_results.csv: {missing_columns}"
    )

# Convert percentage strings into numeric values if necessary
for column in ["Precision", "Recall", "F1-score"]:
    df[column] = (
        df[column]
        .astype(str)
        .str.replace("%", "", regex=False)
        .astype(float)
    )

# Create output directory if needed
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Chart 1: Precision, recall, and F1-score
plt.figure(figsize=(10, 6))

for column in ["Precision", "Recall", "F1-score"]:
    plt.plot(
        df["Threshold"],
        df[column],
        marker="o",
        linewidth=2,
        label=column,
    )

plt.title("Spam Detection Performance by Threshold")
plt.xlabel("Spam Probability Threshold")
plt.ylabel("Score (%)")
plt.xticks(df["Threshold"])
plt.ylim(0, 105)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()

chart_path = OUTPUT_DIR / "threshold_performance.png"
plt.savefig(chart_path, dpi=300)
plt.close()

# Chart 2: False positives and missed spam
plt.figure(figsize=(10, 6))

plt.plot(
    df["Threshold"],
    df["False Positives"],
    marker="o",
    linewidth=2,
    label="False Positives",
)

plt.plot(
    df["Threshold"],
    df["Missed Spam"],
    marker="o",
    linewidth=2,
    label="Missed Spam",
)

plt.title("False Positives vs Missed Spam")
plt.xlabel("Spam Probability Threshold")
plt.ylabel("Number of Messages")
plt.xticks(df["Threshold"])
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()

error_chart_path = OUTPUT_DIR / "threshold_errors.png"
plt.savefig(error_chart_path, dpi=300)
plt.close()

print("\nThreshold charts generated successfully!")
print(f"Performance chart: {chart_path}")
print(f"Error comparison chart: {error_chart_path}")