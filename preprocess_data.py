import os

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder


INPUT_PATH = os.path.join("data", "heart_disease_synthetic.csv")
OUTPUT_PATH = os.path.join("data", "heart_disease_processed.csv")
ENCODERS_PATH = os.path.join("models", "label_encoders.joblib")

CATEGORICAL_COLUMNS = [
    "gender",
    "cholesterol",
    "glucose",
    "blood_sugar",
    "smoking",
    "drinking",
    "yoga",
    "exercise",
    "gym",
]


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    # Fill missing values: median for numeric, mode for categorical.
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and df[col].isna().any():
            mode_value = df[col].mode(dropna=True)
            if not mode_value.empty:
                df[col] = df[col].fillna(mode_value.iloc[0])

    # Derive BMI from weight (kg) and height (cm).
    height_m = df["height"] / 100.0
    df["BMI"] = (df["weight"] / (height_m * height_m)).round(2)

    # Label-encode required categorical columns.
    encoders = {}
    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required categorical column: {col}")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)
    joblib.dump(encoders, ENCODERS_PATH)

    print(f"Processed dataset saved to: {OUTPUT_PATH}")
    print(f"Encoders saved to: {ENCODERS_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Total missing values after cleaning: {int(df.isna().sum().sum())}")


if __name__ == "__main__":
    main()
