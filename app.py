import os
from io import BytesIO

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from predict_heart_disease import predict_heart_disease_risk
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def classify_blood_pressure(systolic: int, diastolic: int) -> tuple[str, str, str]:
    if systolic > 180 or diastolic > 120:
        return (
            "Hypertensive Crisis",
            "Seek urgent medical attention, especially if you have symptoms.",
            "error",
        )
    if systolic >= 140 or diastolic >= 90:
        return (
            "High Blood Pressure (Stage 2)",
            "This BP range is high and should be medically reviewed.",
            "warning",
        )
    if systolic >= 130 or diastolic >= 80:
        return (
            "High Blood Pressure (Stage 1)",
            "Lifestyle changes and clinical follow-up are recommended.",
            "warning",
        )
    if systolic >= 120 and diastolic < 80:
        return (
            "Elevated",
            "BP is above normal. Lifestyle improvement is recommended.",
            "info",
        )
    return (
        "Normal",
        "BP is in the normal range.",
        "success",
    )


def combined_risk_level(model_probability: float, bp_status: str) -> tuple[str, float]:
    adjustment = {
        "Normal": 0.0,
        "Elevated": 0.05,
        "High Blood Pressure (Stage 1)": 0.10,
        "High Blood Pressure (Stage 2)": 0.20,
        "Hypertensive Crisis": 0.30,
    }
    combined_probability = min(1.0, model_probability + adjustment.get(bp_status, 0.0))

    if combined_probability < 0.20:
        return "Low", combined_probability
    if combined_probability < 0.50:
        return "Moderate", combined_probability
    return "High", combined_probability


st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.title("Heart Disease Prediction using Random Forest")

page = st.sidebar.radio("Choose page", ["Train Model", "Risk Calculator"])

if page == "Train Model":
    st.write("Upload a CSV file, select the target column, train the model, and export it.")

    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

    if uploaded_file is None:
        st.info("Please upload a dataset to continue.")
        st.stop()

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read CSV file: {exc}")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), width="stretch")
    st.write(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    if df.empty:
        st.error("The uploaded dataset is empty.")
        st.stop()

    possible_targets = ["target", "output", "heart_disease", "disease", "label"]
    default_idx = 0
    for i, name in enumerate(df.columns):
        if str(name).strip().lower() in possible_targets:
            default_idx = i
            break

    target_column = st.selectbox("Select target column", options=df.columns.tolist(), index=default_idx)

    test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
    n_estimators = st.slider("Number of trees (n_estimators)", min_value=50, max_value=500, value=200, step=50)

    if st.button("Train Random Forest"):
        work_df = df.copy()

        if target_column not in work_df.columns:
            st.error("Selected target column is not in the dataset.")
            st.stop()

        X = work_df.drop(columns=[target_column])
        y = work_df[target_column]

        if X.empty:
            st.error("No feature columns available after removing target.")
            st.stop()

        X = pd.get_dummies(X, drop_first=True)

        if y.dtype == "object":
            y = y.astype("category").cat.codes

        if len(np.unique(y)) < 2:
            st.error("Target column must contain at least two classes.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=int(random_state),
            stratify=y,
        )

        model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            random_state=int(random_state),
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.subheader("Model Performance")
        st.metric("Accuracy", f"{acc:.4f}")

        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, width="stretch")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(x=importances.values, y=importances.index, ax=ax2)
        ax2.set_title("Top Feature Importances")
        ax2.set_xlabel("Importance")
        ax2.set_ylabel("Feature")
        st.pyplot(fig2)

        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "random_forest_heart_disease.pkl")
        joblib.dump(model, model_path)
        st.success(f"Model saved to {model_path}")

        model_buffer = BytesIO()
        joblib.dump(model, model_buffer)
        model_buffer.seek(0)

        st.download_button(
            label="Download model (.pkl)",
            data=model_buffer,
            file_name="random_forest_heart_disease.pkl",
            mime="application/octet-stream",
        )
else:
    st.write("Enter patient details to calculate heart disease risk.")

    with st.form("risk_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=250, value=120, step=1)
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80, step=1)
            height = st.number_input("Height (cm)", min_value=80.0, max_value=250.0, value=170.0, step=0.1)
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=75.0, step=0.1)
            cholesterol = st.selectbox("Cholesterol", ["Normal", "Low", "High"])
            glucose = st.selectbox("Glucose", ["Normal", "Low", "High"])

        with col2:
            blood_sugar = st.selectbox("Blood Sugar", ["Normal", "Low", "High"])
            smoking = st.selectbox("Smoking", ["No", "Yes"])
            drinking = st.selectbox("Drinking", ["No", "Yes"])
            yoga = st.selectbox("Yoga", ["No", "Yes"])
            exercise = st.selectbox("Exercise", ["No", "Yes"])
            gym = st.selectbox("Gym", ["No", "Yes"])

        submitted = st.form_submit_button("Calculate Risk")

    if submitted:
        user_input = {
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "cholesterol": cholesterol,
            "glucose": glucose,
            "blood_sugar": blood_sugar,
            "smoking": smoking,
            "drinking": drinking,
            "yoga": yoga,
            "exercise": exercise,
            "gym": gym,
        }

        try:
            result = predict_heart_disease_risk(user_input)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.info("Run your training pipeline first so model and preprocessing artifacts are available.")
        except Exception as exc:
            st.error(f"Could not calculate risk: {exc}")
        else:
            probability = result["probability_heart_disease"]
            bp_status, bp_note, bp_message_type = classify_blood_pressure(int(systolic_bp), int(diastolic_bp))
            overall_level, combined_probability = combined_risk_level(probability, bp_status)

            st.subheader("Risk Result")
            st.metric("Heart disease probability", f"{probability * 100:.2f}%")
            st.success(result["risk_label"])

            st.subheader("Blood Pressure Analysis")
            st.metric("Blood Pressure", f"{int(systolic_bp)}/{int(diastolic_bp)} mmHg")
            st.metric("BP Category", bp_status)

            if bp_message_type == "error":
                st.error(bp_note)
            elif bp_message_type == "warning":
                st.warning(bp_note)
            elif bp_message_type == "info":
                st.info(bp_note)
            else:
                st.success(bp_note)

            st.subheader("Overall Interpretation")
            st.metric("Combined risk estimate", f"{combined_probability * 100:.2f}%")
            if overall_level == "High":
                st.error(f"Overall risk level: {overall_level}")
            elif overall_level == "Moderate":
                st.warning(f"Overall risk level: {overall_level}")
            else:
                st.success(f"Overall risk level: {overall_level}")

            st.caption(
                "Note: The ML model prediction is based on trained dataset features. "
                "Blood pressure is shown as an added clinical-style analysis layer and is not a medical diagnosis."
            )
