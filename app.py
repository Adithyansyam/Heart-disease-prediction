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

st.sidebar.markdown("## CardioInsight")
st.sidebar.caption("Heart disease screening workspace")
selected_theme = st.sidebar.selectbox("Visual Theme", options=list(THEMES.keys()), index=0)
theme = THEMES[selected_theme]

theme_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg-1: __BG_1__;
        --bg-2: __BG_2__;
        --ink: __INK__;
        --muted: __MUTED__;
        --brand: __BRAND__;
        --brand-2: __BRAND_2__;
        --accent: __ACCENT__;
        --surface: __SURFACE__;
        --border: __BORDER__;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 8% 8%, rgba(42, 157, 143, 0.16) 0%, transparent 35%),
            radial-gradient(circle at 92% 12%, rgba(255, 127, 80, 0.16) 0%, transparent 32%),
            linear-gradient(180deg, var(--bg-1), var(--bg-2));
        color: var(--ink);
        font-family: 'Space Grotesk', sans-serif;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(255, 255, 255, 0.24);
        background: __SIDEBAR_GRAD__;
    }

    [data-testid="stSidebar"] * {
        color: #e9f5ff;
    }

    .hero {
        margin: 0 0 1.2rem 0;
        padding: 1.3rem 1.5rem;
        border-radius: 18px;
        border: 1px solid var(--border);
        background: linear-gradient(120deg, rgba(0, 109, 119, 0.90), rgba(42, 157, 143, 0.88));
        color: #f4fdff;
        box-shadow: 0 14px 35px rgba(12, 49, 71, 0.20);
        animation: fadein 420ms ease;
    }

    .hero h1 {
        margin: 0;
        letter-spacing: 0.2px;
        font-size: clamp(1.4rem, 2.4vw, 2.0rem);
        font-weight: 700;
    }

    .hero p {
        margin: 0.35rem 0 0 0;
        opacity: 0.95;
        font-size: 0.98rem;
    }

    .hero-meta {
        margin-top: 0.7rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
    }

    .hero-pill {
        border: 1px solid rgba(255, 255, 255, 0.35);
        border-radius: 999px;
        padding: 0.18rem 0.65rem;
        font-size: 0.72rem;
        letter-spacing: 0.25px;
        background: rgba(255, 255, 255, 0.12);
    }

    .glass-card {
        border-radius: 16px;
        border: 1px solid var(--border);
        background: var(--surface);
        backdrop-filter: blur(8px);
        padding: 0.85rem 1rem;
        box-shadow: 0 10px 22px rgba(16, 31, 52, 0.08);
        margin-bottom: 0.9rem;
        animation: rise 430ms ease;
    }

    .metric-chip {
        border-radius: 14px;
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.82);
        padding: 0.9rem 1rem;
        min-height: 102px;
        box-shadow: 0 8px 18px rgba(13, 39, 68, 0.08);
    }

    .metric-chip .label {
        font-size: 0.78rem;
        color: var(--muted);
        letter-spacing: 0.35px;
        text-transform: uppercase;
        margin-bottom: 0.15rem;
    }

    .metric-chip .value {
        font-size: clamp(1.45rem, 1.9vw, 1.9rem);
        font-weight: 700;
        color: var(--ink);
        line-height: 1.2;
    }

    .badge {
        display: inline-block;
        margin-top: 0.35rem;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.25px;
    }

    .badge.low { background: rgba(42, 157, 143, 0.18); color: #15695f; }
    .badge.moderate { background: rgba(249, 199, 79, 0.28); color: #7e5a00; }
    .badge.high { background: rgba(231, 111, 81, 0.2); color: #8f2313; }

    .section-title {
        margin: 0.9rem 0 0.45rem 0;
        color: #12314a;
        font-weight: 700;
        font-size: 1.03rem;
    }

    .insight-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.7rem;
        margin: 0.35rem 0 1rem 0;
    }

    .insight-card {
        border-radius: 14px;
        border: 1px solid var(--border);
        padding: 0.7rem 0.8rem;
        background: rgba(255, 255, 255, 0.80);
        box-shadow: 0 8px 18px rgba(13, 39, 68, 0.08);
    }

    .insight-label {
        font-size: 0.76rem;
        color: var(--muted);
        letter-spacing: 0.28px;
        text-transform: uppercase;
    }

    .insight-value {
        margin-top: 0.1rem;
        color: var(--ink);
        font-size: 1.05rem;
        font-weight: 700;
    }

    .subtle {
        color: var(--muted);
        font-size: 0.92rem;
        margin-bottom: 0.65rem;
    }

    .stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] button {
        border-radius: 12px;
        border: none;
        font-weight: 600;
        padding: 0.55rem 1rem;
        color: #f6fbff;
        background: linear-gradient(120deg, #0f6e7f 0%, #1f9ea2 100%);
        box-shadow: 0 8px 16px rgba(9, 68, 79, 0.24);
        transition: transform 0.18s ease, filter 0.18s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-1px);
        filter: brightness(1.05);
    }

    label, .stMarkdown, .stCaption, p, h1, h2, h3, h4 {
        color: var(--ink);
    }

    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span,
    [data-testid="stWidgetLabel"] label {
        color: var(--ink) !important;
        font-weight: 600;
    }

    [data-baseweb="input"] > div,
    [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input {
        border-radius: 10px !important;
        border-color: rgba(17, 40, 64, 0.2) !important;
        background: rgba(255, 255, 255, 0.82) !important;
        color: var(--ink) !important;
    }

    [data-baseweb="select"] div,
    [data-baseweb="input"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        color: var(--ink) !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: color-mix(in srgb, var(--muted) 75%, white) !important;
        opacity: 1 !important;
    }

    [data-testid="stTabs"] [role="tablist"] {
        gap: 8px;
    }

    [data-testid="stTabs"] [role="tab"] {
        border-radius: 999px;
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.72);
        padding: 0.35rem 0.9rem;
        font-weight: 600;
    }

    [data-testid="stTabs"] [aria-selected="true"] {
        background: linear-gradient(120deg, var(--brand), var(--brand-2));
        color: #f7fcff;
        border-color: transparent;
    }

    @keyframes fadein {
        from {opacity: 0; transform: translateY(6px);} to {opacity: 1; transform: translateY(0);} }
    @keyframes rise {
        from {opacity: 0; transform: translateY(8px);} to {opacity: 1; transform: translateY(0);} }

    @media (max-width: 900px) {
        .hero { padding: 1rem 1.05rem; }
        .metric-chip { min-height: 92px; }
    }
    </style>
    """

for key, value in theme.items():
    theme_css = theme_css.replace(f"__{key.upper()}__", value)

st.markdown(theme_css, unsafe_allow_html=True)


def render_hero(title: str, subtitle: str, pills: list[str] | None = None) -> None:
    pills = pills or []
    pills_html = "".join([f'<span class="hero-pill">{pill}</span>' for pill in pills])
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <div class="hero-meta">{pills_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_chip(label: str, value: str, badge_text: str | None = None, badge_tone: str = "low") -> None:
    badge_html = f'<span class="badge {badge_tone}">{badge_text}</span>' if badge_text else ""
    st.markdown(
        f"""
        <div class="metric-chip">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight_cards(cards: list[tuple[str, str]]) -> None:
    card_html = ""
    for label, value in cards:
        card_html += (
            '<div class="insight-card">'
            f'<div class="insight-label">{label}</div>'
            f'<div class="insight-value">{value}</div>'
            "</div>"
        )
    st.markdown(f'<div class="insight-grid">{card_html}</div>', unsafe_allow_html=True)


def bmi_category(height_cm: float, weight_kg: float) -> tuple[float, str]:
    meters = height_cm / 100.0
    bmi = weight_kg / (meters * meters)
    if bmi < 18.5:
        return bmi, "Underweight"
    if bmi < 25:
        return bmi, "Normal"
    if bmi < 30:
        return bmi, "Overweight"
    return bmi, "Obese"


page = st.sidebar.radio("Workspace", ["Risk Calculator", "Train Model"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Small clinical decision support UI for model exploration and personal risk estimation."
)
st.sidebar.caption("Design mode: " + selected_theme)

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
            activity = st.selectbox("Activity (Yoga / Exercise)", ["No", "Yes"])

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
            active_habits = int(activity == "Yes")
            risk_habits = sum([smoking == "Yes", drinking == "Yes"])
            progress.progress(100, text="Dashboard ready")
            time.sleep(0.06)

            st.markdown('<p class="section-title">Risk Dashboard</p>', unsafe_allow_html=True)
            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            with kpi_col1:
                render_metric_chip("Heart disease probability", f"{probability * 100:.2f}%")
            with kpi_col2:
                render_metric_chip("Prediction confidence", f"{confidence * 100:.2f}%")
            with kpi_col3:
                tone = "high" if overall_level == "High" else "moderate" if overall_level == "Moderate" else "low"
                render_metric_chip(
                    "Combined risk estimate",
                    f"{combined_probability * 100:.2f}%",
                    badge_text=overall_level,
                    badge_tone=tone,
                )

            if prediction == 1:
                st.error(
                    f"Risk class: {result['risk_label']}. Model predicts approximately {probability * 100:.2f}% chance of heart disease."
                )
            else:
                st.success(
                    f"Risk class: {result['risk_label']}. Model predicts approximately {probability * 100:.2f}% chance of heart disease."
                )

            if activity == "Yes":
                st.info(
                    f"Activity benefit applied: base risk reduced by 10% ({base_probability * 100:.2f}% to {activity_adjusted_probability * 100:.2f}%)."
                )

            if low_penalty > 0:
                st.warning(
                    f"Low-value adjustment applied: +{low_penalty * 100:.0f}% risk due to {low_count} low selection(s) in cholesterol/glucose/blood sugar."
                )

            render_insight_cards(
                [
                    ("BMI", f"{bmi:.1f} ({bmi_status})"),
                    ("Active habits", f"{active_habits} / 1"),
                    ("Risk habits", f"{risk_habits} / 2"),
                    ("Blood pressure", f"{int(systolic_bp)}/{int(diastolic_bp)} mmHg"),
                ]
            )

            tab_overview, tab_clinical, tab_reliability = st.tabs(["Overview", "Clinical Detail", "Reliability"])

            with tab_overview:
                st.markdown('<p class="subtle">Combined probability meter</p>', unsafe_allow_html=True)
                st.progress(int(combined_probability * 100), text=f"Risk Meter: {combined_probability * 100:.2f}%")
                fig_gauge, ax_gauge = plt.subplots(figsize=(4.8, 2.8))
                wedges = [combined_probability, 1.0 - combined_probability]
                colors = ["#e76f51" if combined_probability >= 0.5 else "#f4a261", "#e5edf6"]
                ax_gauge.pie(
                    wedges,
                    colors=colors,
                    startangle=180,
                    counterclock=False,
                    wedgeprops={"width": 0.35, "edgecolor": "white"},
                )
                ax_gauge.text(0, -0.05, f"{combined_probability * 100:.1f}%", ha="center", va="center", fontsize=16, fontweight="bold")
                ax_gauge.text(0, -0.28, "Integrated Risk", ha="center", va="center", fontsize=9)
                ax_gauge.set_aspect("equal")
                st.pyplot(fig_gauge)

            with tab_clinical:
                st.markdown('<p class="section-title">Blood Pressure Analysis</p>', unsafe_allow_html=True)
                st.write(f"Measured BP: {int(systolic_bp)}/{int(diastolic_bp)} mmHg")
                st.write(f"Category: {bp_status}")
                if bp_message_type == "error":
                    st.error(bp_note)
                elif bp_message_type == "warning":
                    st.warning(bp_note)
                elif bp_message_type == "info":
                    st.info(bp_note)
                else:
                    st.success(bp_note)
                if bmi_status in {"Overweight", "Obese"}:
                    st.warning("BMI is above the normal range. Weight management can lower cardiovascular risk.")
                elif bmi_status == "Underweight":
                    st.info("BMI is below the normal range. Nutritional assessment may be helpful.")
                else:
                    st.success("BMI is within the normal range.")

            with tab_reliability:
                st.markdown('<p class="section-title">Model Reliability</p>', unsafe_allow_html=True)
                try:
                    model_acc = get_saved_model_accuracy()
                except Exception as exc:
                    st.info(f"Model test accuracy unavailable: {exc}")
                    model_acc = None
                else:
                    acc_pct = model_acc * 100
                    render_metric_chip("Test accuracy", f"{acc_pct:.2f}%")
                    if acc_pct >= 90:
                        st.success("Model reliability appears strong on held-out test data.")
                    elif acc_pct >= 75:
                        st.info("Model reliability appears acceptable on held-out test data.")
                    else:
                        st.warning("Model reliability is modest. Treat predictions cautiously.")

            st.markdown('<p class="section-title">Interpretation</p>', unsafe_allow_html=True)
            if overall_level == "High":
                st.error(f"Overall risk level: {overall_level} ({combined_probability * 100:.2f}%)")
            elif overall_level == "Moderate":
                st.warning(f"Overall risk level: {overall_level} ({combined_probability * 100:.2f}%)")
            else:
                st.success(f"Overall risk level: {overall_level} ({combined_probability * 100:.2f}%)")

            st.caption(
                "This output is generated from a machine learning model and blood pressure logic. "
                "It supports awareness and is not a medical diagnosis."
            )
