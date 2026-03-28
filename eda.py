import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


INPUT_PATH = os.path.join("data", "heart_disease_processed.csv")
ENCODERS_PATH = os.path.join("models", "label_encoders.joblib")
OUTPUT_DIR = os.path.join("outputs", "eda")


def maybe_decode(series: pd.Series, column_name: str, encoders: dict) -> pd.Series:
    if column_name not in encoders:
        return series
    encoder = encoders[column_name]
    decoded = encoder.inverse_transform(series.astype(int))
    return pd.Series(decoded, index=series.index)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = pd.read_csv(INPUT_PATH)
    encoders = {}
    if os.path.exists(ENCODERS_PATH):
        encoders = joblib.load(ENCODERS_PATH)

    # 1) Class distribution plot
    plt.figure(figsize=(7, 5))
    class_counts = df["heart_disease"].value_counts().sort_index()
    ax = sns.barplot(x=class_counts.index.astype(str), y=class_counts.values, color="#72b7b2")
    ax.set_title("Class Distribution: heart_disease")
    ax.set_xlabel("heart_disease")
    ax.set_ylabel("Count")
    for idx, count in enumerate(class_counts.values):
        ax.text(idx, count + 5, str(count), ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution_heart_disease.png"), dpi=200)
    plt.close()

    # 2) Correlation heatmap for all features
    plt.figure(figsize=(12, 9))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True)
    plt.title("Correlation Heatmap (All Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=200)
    plt.close()

    # 3) Individual feature distribution plots
    # Age distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["age"], kde=True, bins=20, color="#1f77b4")
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "distribution_age.png"), dpi=200)
    plt.close()

    # BMI distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["BMI"], kde=True, bins=20, color="#2ca02c")
    plt.title("BMI Distribution")
    plt.xlabel("BMI")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "distribution_bmi.png"), dpi=200)
    plt.close()

    # Cholesterol distribution (decoded labels if encoders are available)
    cholesterol_series = maybe_decode(df["cholesterol"], "cholesterol", encoders)
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x=cholesterol_series, order=["Low", "Medium", "High"], color="#8ecae6")
    ax.set_title("Cholesterol Distribution")
    ax.set_xlabel("Cholesterol")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "distribution_cholesterol.png"), dpi=200)
    plt.close()

    # Glucose distribution (decoded labels if encoders are available)
    glucose_series = maybe_decode(df["glucose"], "glucose", encoders)
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x=glucose_series, order=["Low", "Medium", "High"], color="#bde0fe")
    ax.set_title("Glucose Distribution")
    ax.set_xlabel("Glucose")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "distribution_glucose.png"), dpi=200)
    plt.close()

    print("EDA completed. Plots saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
