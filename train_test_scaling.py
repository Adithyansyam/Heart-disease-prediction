import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


INPUT_PATH = os.path.join("data", "heart_disease_processed.csv")
TARGET_COLUMN = "heart_disease"
NUMERICAL_FEATURES = ["age", "height", "weight", "BMI"]

X_TRAIN_PATH = os.path.join("data", "X_train.csv")
X_TEST_PATH = os.path.join("data", "X_test.csv")
Y_TRAIN_PATH = os.path.join("data", "y_train.csv")
Y_TEST_PATH = os.path.join("data", "y_test.csv")
SCALER_PATH = os.path.join("models", "standard_scaler.joblib")


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    for feature in NUMERICAL_FEATURES:
        if feature not in X.columns:
            raise ValueError(f"Required numerical feature '{feature}' not found in X.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled_num = pd.DataFrame(
        scaler.fit_transform(X_train[NUMERICAL_FEATURES]),
        columns=NUMERICAL_FEATURES,
        index=X_train.index,
    )
    X_test_scaled_num = pd.DataFrame(
        scaler.transform(X_test[NUMERICAL_FEATURES]),
        columns=NUMERICAL_FEATURES,
        index=X_test.index,
    )

    X_train_non_num = X_train.drop(columns=NUMERICAL_FEATURES)
    X_test_non_num = X_test.drop(columns=NUMERICAL_FEATURES)

    X_train = pd.concat([X_train_non_num, X_train_scaled_num], axis=1)[X.columns]
    X_test = pd.concat([X_test_non_num, X_test_scaled_num], axis=1)[X.columns]

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    X_train.to_csv(X_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    y_train.to_frame(name=TARGET_COLUMN).to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_frame(name=TARGET_COLUMN).to_csv(Y_TEST_PATH, index=False)
    joblib.dump(scaler, SCALER_PATH)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print("Scaled numerical features:", ", ".join(NUMERICAL_FEATURES))
    print(f"Scaler saved to: {SCALER_PATH}")


if __name__ == "__main__":
    main()
