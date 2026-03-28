import os
from typing import Any, Dict

import joblib
import pandas as pd


MODEL_PATH = os.path.join("models", "random_forest_model.joblib")
MODEL_EXPORT_PATH = os.path.join("models", "random_forest_model_step10.joblib")
SCALER_PATH = os.path.join("models", "standard_scaler.joblib")
ENCODERS_PATH = os.path.join("models", "label_encoders.joblib")
X_TRAIN_PATH = os.path.join("data", "X_train.csv")

NUMERICAL_COLUMNS = ["age", "height", "weight", "BMI"]
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


def _canonicalize_value(column: str, value: Any) -> str:
    text = str(value).strip().lower()

    if column == "gender":
        mapping = {
            "m": "Male",
            "male": "Male",
            "f": "Female",
            "female": "Female",
        }
        if text not in mapping:
            raise ValueError("gender must be Male or Female")
        return mapping[text]

    if column in ["cholesterol", "glucose", "blood_sugar"]:
        mapping = {
            "low": "Low",
            "normal": "Medium",
            "medium": "Medium",
            "med": "Medium",
            "high": "High",
        }
        if text not in mapping:
            raise ValueError(f"{column} must be Normal, Low, or High")
        return mapping[text]

    if column in ["smoking", "drinking", "yoga", "exercise", "gym"]:
        mapping = {
            "yes": "Yes",
            "y": "Yes",
            "true": "Yes",
            "1": "Yes",
            "no": "No",
            "n": "No",
            "false": "No",
            "0": "No",
        }
        if text not in mapping:
            raise ValueError(f"{column} must be Yes or No")
        return mapping[text]

    return str(value)


def _preprocess_input(
    user_input: Dict[str, Any],
    encoders: Dict[str, Any],
    scaler: Any,
    feature_columns: list,
) -> pd.DataFrame:
    required_fields = [
        "age",
        "gender",
        "height",
        "weight",
        "cholesterol",
        "glucose",
        "blood_sugar",
        "smoking",
        "drinking",
        "yoga",
        "exercise",
        "gym",
    ]

    missing_fields = [field for field in required_fields if field not in user_input]
    if missing_fields:
        raise ValueError(f"Missing required input fields: {missing_fields}")

    row = {
        "age": float(user_input["age"]),
        "gender": _canonicalize_value("gender", user_input["gender"]),
        "height": float(user_input["height"]),
        "weight": float(user_input["weight"]),
        "cholesterol": _canonicalize_value("cholesterol", user_input["cholesterol"]),
        "glucose": _canonicalize_value("glucose", user_input["glucose"]),
        "blood_sugar": _canonicalize_value("blood_sugar", user_input["blood_sugar"]),
        "smoking": _canonicalize_value("smoking", user_input["smoking"]),
        "drinking": _canonicalize_value("drinking", user_input["drinking"]),
        "yoga": _canonicalize_value("yoga", user_input["yoga"]),
        "exercise": _canonicalize_value("exercise", user_input["exercise"]),
        "gym": _canonicalize_value("gym", user_input["gym"]),
    }

    if row["height"] <= 0:
        raise ValueError("height must be greater than 0")

    height_m = row["height"] / 100.0
    row["BMI"] = round(row["weight"] / (height_m * height_m), 2)

    df = pd.DataFrame([row])

    # Apply the same label encoding used during training.
    for col in CATEGORICAL_COLUMNS:
        if col not in encoders:
            raise ValueError(f"Encoder not found for column: {col}")
        df[col] = encoders[col].transform(df[col])

    # Apply the same numerical scaling used during training.
    df_scaled_num = pd.DataFrame(
        scaler.transform(df[NUMERICAL_COLUMNS]),
        columns=NUMERICAL_COLUMNS,
        index=df.index,
    )
    df_non_num = df.drop(columns=NUMERICAL_COLUMNS)
    df = pd.concat([df_non_num, df_scaled_num], axis=1)

    # Ensure strict feature order expected by the model.
    df = df[feature_columns]
    return df


def predict_heart_disease_risk(user_input: Dict[str, Any]) -> Dict[str, Any]:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    feature_columns = pd.read_csv(X_TRAIN_PATH, nrows=1).columns.tolist()

    input_df = _preprocess_input(user_input, encoders, scaler, feature_columns)

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    risk_label = "Heart disease risk" if prediction == 1 else "No heart disease risk"
    return {
        "prediction": prediction,
        "risk_label": risk_label,
        "probability_heart_disease": round(probability, 4),
    }


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Run train_model.py first to save the model using Joblib."
        )

    # Demonstration: save trained model with Joblib (Step 10 requirement).
    model = joblib.load(MODEL_PATH)
    joblib.dump(model, MODEL_EXPORT_PATH)

    # Demonstration: load saved model and predict for a new user profile.
    sample_user = {
        "age": 54,
        "gender": "Male",
        "height": 170,
        "weight": 82,
        "cholesterol": "High",
        "glucose": "Medium",
        "blood_sugar": "High",
        "smoking": "Yes",
        "drinking": "No",
        "yoga": "No",
        "exercise": "No",
        "gym": "No",
    }

    result = predict_heart_disease_risk(sample_user)

    print(f"Model saved with Joblib to: {MODEL_EXPORT_PATH}")
    print("Model loaded successfully from Joblib file.")
    print("Prediction demo for new user input:")
    print(sample_user)
    print(result)


if __name__ == "__main__":
    main()
