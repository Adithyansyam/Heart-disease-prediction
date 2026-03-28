import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


X_TRAIN_PATH = os.path.join("data", "X_train.csv")
Y_TRAIN_PATH = os.path.join("data", "y_train.csv")
MODEL_PATH = os.path.join("models", "random_forest_model.joblib")


def main() -> None:
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)["heart_disease"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("RandomForestClassifier trained successfully.")
    print("n_estimators=100, random_state=42")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
