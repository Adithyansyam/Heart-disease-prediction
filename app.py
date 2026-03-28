import os
from io import BytesIO
import time

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


@st.cache_data(show_spinner=False)
def get_saved_model_accuracy() -> float:
    model_path = os.path.join("models", "random_forest_model.joblib")
    x_test_path = os.path.join("data", "X_test.csv")
    y_test_path = os.path.join("data", "y_test.csv")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(x_test_path):
        raise FileNotFoundError(f"Test feature file not found: {x_test_path}")
    if not os.path.exists(y_test_path):
        raise FileNotFoundError(f"Test target file not found: {y_test_path}")

    model = joblib.load(model_path)
    X_test = pd.read_csv(x_test_path)
    y_test_df = pd.read_csv(y_test_path)

    if "heart_disease" in y_test_df.columns:
        y_test = y_test_df["heart_disease"]
    else:
        y_test = y_test_df.iloc[:, 0]

    y_pred = model.predict(X_test)
    return float(accuracy_score(y_test, y_pred))


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


def bmi_category(height_cm: float, weight_kg: float) -> tuple[float, str]:
    height_m = max(height_cm / 100.0, 0.01)
    bmi = weight_kg / (height_m**2)
    if bmi < 18.5:
        return bmi, "Underweight"
    if bmi < 25.0:
        return bmi, "Normal"
    if bmi < 30.0:
        return bmi, "Overweight"
    return bmi, "Obese"


def render_hero(title: str, subtitle: str, pills: list[str] | None = None) -> None:
    st.title(title)
    st.caption(subtitle)
    if pills:
        st.write(" | ".join(pills))


def render_metric_chip(label: str, value: str, badge_text: str | None = None, badge_tone: str | None = None) -> None:
    tone_prefix = ""
    if badge_text:
        tone_prefix = f" [{badge_tone}]" if badge_tone else ""
        badge = f" ({badge_text}{tone_prefix})"
    else:
        badge = ""
    st.metric(label, f"{value}{badge}")


def render_insight_cards(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(f"**{label}**")
            st.write(value)


st.set_page_config(page_title="CardioInsight | Heart Risk Intelligence", layout="wide")

THEMES = {
    "Teal Horizon": {
        "bg_1": "#f4f9ff",
        "bg_2": "#eef6ff",
        "ink": "#0f2338",
        "muted": "#51647a",
        "brand": "#006d77",
        "brand_2": "#2a9d8f",
        "accent": "#ff7f50",
        "surface": "rgba(255, 255, 255, 0.74)",
        "border": "rgba(15, 35, 56, 0.12)",
        "sidebar_grad": "linear-gradient(160deg, #0f2338 0%, #133450 45%, #1b4965 100%)",
    },
    "Sunset Clinic": {
        "bg_1": "#fff9f2",
        "bg_2": "#fff2e8",
        "ink": "#2f1e1a",
        "muted": "#6b4c45",
        "brand": "#bc4749",
        "brand_2": "#dd6b4d",
        "accent": "#386641",
        "surface": "rgba(255, 255, 255, 0.72)",
        "border": "rgba(47, 30, 26, 0.14)",
        "sidebar_grad": "linear-gradient(160deg, #532c28 0%, #7a3e35 45%, #a44d3f 100%)",
    },
}

st.sidebar.title("CardioInsight")
st.sidebar.write("Heart disease screening workspace")
page = st.sidebar.radio("Workspace", ["Risk Calculator", "Train Model"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Small clinical decision support UI for model exploration and personal risk estimation."
)

if page == "Train Model":
    render_hero(
        "Model Lab",
        "Upload a dataset, train a Random Forest model, inspect evaluation outputs, and export the trained artifact.",
        pills=["Random Forest", "Visual Evaluation", "One-Click Export"],
    )

    uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV file to begin model training.")
        st.stop()

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read CSV file: {exc}")
        st.stop()

    st.markdown('<p class="section-title">Dataset Snapshot</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtle">Quick look at records to validate structure before training.</p>', unsafe_allow_html=True)
    st.dataframe(df.head(), width="stretch")
    st.caption(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    if df.empty:
        st.error("The uploaded dataset is empty.")
        st.stop()

    possible_targets = ["target", "output", "heart_disease", "disease", "label"]
    default_idx = 0
    for i, name in enumerate(df.columns):
        if str(name).strip().lower() in possible_targets:
            default_idx = i
            break

    st.markdown('<p class="section-title">Training Controls</p>', unsafe_allow_html=True)
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    with ctrl_col1:
        target_column = st.selectbox("Target column", options=df.columns.tolist(), index=default_idx)
    with ctrl_col2:
        test_size = st.slider("Test split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    with ctrl_col3:
        n_estimators = st.slider("Trees", min_value=50, max_value=500, value=200, step=50)
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

    if st.button("Train Random Forest"):
        with st.spinner("Training model and preparing evaluation visuals..."):
            progress = st.progress(0, text="Preparing dataset")
            time.sleep(0.08)
            progress.progress(25, text="Encoding features")

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
        progress.progress(45, text="Splitting train and test sets")

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
        progress.progress(78, text="Computing evaluation metrics")

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        progress.progress(100, text="Done")
        time.sleep(0.06)

        st.markdown('<p class="section-title">Model Performance</p>', unsafe_allow_html=True)
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        with perf_col1:
            render_metric_chip("Accuracy", f"{acc:.4f}")
        with perf_col2:
            render_metric_chip("Train rows", f"{len(X_train):,}")
        with perf_col3:
            render_metric_chip("Test rows", f"{len(X_test):,}")

        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, width="stretch")

        vis_col1, vis_col2 = st.columns(2)
        with vis_col1:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
        with vis_col2:
            importances = (
                pd.Series(model.feature_importances_, index=X.columns)
                .sort_values(ascending=False)
                .head(15)
            )
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
    render_hero(
        "Heart Risk Calculator",
        "Enter a patient profile to estimate heart disease probability and get an interpretable BP-aware risk summary.",
        pills=["Preventive Focus", "Clinical-Style Summary", "Fast Inference"],
    )

    with st.form("risk_form"):
        st.markdown('<p class="section-title">Patient Profile</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtle">Fill all fields for a complete prediction.</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=20)
            gender = st.selectbox("Gender", ["Male", "Female"], index=0)
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=250, value=120)
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, value=80)
            height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
            weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=60.0)
            cholesterol = st.selectbox("Cholesterol", ["Normal", "Low", "High"], index=0)
            glucose = st.selectbox("Glucose", ["Normal", "Low", "High"], index=0)

        with col2:
            blood_sugar = st.selectbox("Blood Sugar", ["Normal", "Low", "High"], index=0)
            smoking = st.selectbox("Smoking", ["No", "Yes"], index=0)
            drinking = st.selectbox("Drinking", ["No", "Yes"], index=0)
            activity = st.selectbox("Activity (Yoga / Exercise)", ["No", "Yes"], index=1)

        submitted = st.form_submit_button("Calculate Risk")

    if submitted:
        with st.spinner("Running inference and generating report..."):
            progress = st.progress(0, text="Validating input")
            time.sleep(0.08)

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
            "yoga": activity,
            "exercise": activity,
            "gym": "No",
        }

        try:
            progress.progress(35, text="Loading model artifacts")
            result = predict_heart_disease_risk(user_input)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.info("Run the training pipeline so model and preprocessing artifacts are available.")
        except Exception as exc:
            st.error(f"Could not calculate risk: {exc}")
        else:
            progress.progress(78, text="Calculating integrated risk")
            base_probability = result["probability_heart_disease"]
            prediction = int(result["prediction"])
            activity_adjustment = 0.9 if activity == "Yes" else 1.0
            activity_adjusted_probability = base_probability * activity_adjustment
            low_count = sum([
                cholesterol == "Low",
                glucose == "Low",
                blood_sugar == "Low",
            ])
            low_penalty = min(0.04, 0.02 * low_count)
            probability = min(1.0, activity_adjusted_probability + low_penalty)
            confidence = probability if prediction == 1 else (1.0 - probability)
            bp_status, bp_note, bp_message_type = classify_blood_pressure(int(systolic_bp), int(diastolic_bp))
            overall_level, combined_probability = combined_risk_level(probability, bp_status)
            bmi, bmi_status = bmi_category(float(height), float(weight))
            progress.empty()

            st.markdown('<p class="subtle">Risk meter (combined)</p>', unsafe_allow_html=True)
            st.progress(
                int(combined_probability * 100),
                text=f"Risk Meter: {combined_probability * 100:.2f}% ({overall_level})",
            )

            bmi_progress = int(min(100, max(0, (bmi / 40.0) * 100)))
            st.markdown('<p class="subtle">BMI meter</p>', unsafe_allow_html=True)
            st.progress(
                bmi_progress,
                text=f"BMI: {bmi:.1f} ({bmi_status})",
            )
