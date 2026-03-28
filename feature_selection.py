import os

import pandas as pd


INPUT_PATH = os.path.join("data", "heart_disease_processed.csv")
X_OUTPUT_PATH = os.path.join("data", "X_features.csv")
Y_OUTPUT_PATH = os.path.join("data", "y_target.csv")
TARGET_COLUMN = "heart_disease"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    os.makedirs("data", exist_ok=True)
    X.to_csv(X_OUTPUT_PATH, index=False)
    y.to_frame(name=TARGET_COLUMN).to_csv(Y_OUTPUT_PATH, index=False)

    feature_list = X.columns.tolist()

    print(f"Input features shape (X): {X.shape}")
    print(f"Target shape (y): {y.shape}")
    print("Final list of features used for training:")
    for idx, feature in enumerate(feature_list, start=1):
        print(f"{idx}. {feature}")


if __name__ == "__main__":
    main()
