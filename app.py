import os
from io import BytesIO
import time
import hashlib

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
    chips = "".join([f'<span class="hero-chip">{item}</span>' for item in (pills or [])])
    st.markdown(
        f"""
        <div class="hero-shell">
            <p class="hero-kicker">CardioInsight</p>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-subtitle">{subtitle}</p>
            <div class="hero-chips">{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_chip(label: str, value: str, badge_text: str | None = None, badge_tone: str | None = None) -> None:
    badge_class = "tone-neutral"
    if badge_tone == "high":
        badge_class = "tone-high"
    elif badge_tone == "moderate":
        badge_class = "tone-moderate"
    elif badge_tone == "low":
        badge_class = "tone-low"

    badge_html = ""
    if badge_text:
        badge_html = f'<span class="metric-badge {badge_class}">{badge_text}</span>'

    st.markdown(
        f"""
        <div class="metric-card">
            <p class="metric-label">{label}</p>
            <p class="metric-value">{value}</p>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_meter(label: str, value: float, status_text: str, gradient: str) -> None:
    bounded = max(0.0, min(100.0, value))
    st.markdown(
        f"""
        <div class="meter-card">
            <div class="meter-head">
                <span>{label}</span>
                <span>{bounded:.1f}%</span>
            </div>
            <div class="meter-track">
                <div class="meter-fill" style="width:{bounded}%; background:{gradient};"></div>
            </div>
            <div class="meter-foot">{status_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight_cards(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(f"**{label}**")
            st.write(value)


ALLOWED_EMAILS = {
    "gaurisnair@gmail.com",
    "adithyasunil@gmail.com",
    "ansalna@gmail.com",
    "mabay@gmail.com",
}
PASSWORD_HASH = hashlib.sha256("12345678".encode("utf-8")).hexdigest()


def _check_credentials(email: str, password: str) -> bool:
    clean_email = email.strip().lower()
    password_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return clean_email in ALLOWED_EMAILS and password_hash == PASSWORD_HASH


def _init_auth_state() -> None:
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = ""


def render_login_screen() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .login-card {
            background: linear-gradient(145deg, rgba(255,255,255,0.92), rgba(255,255,255,0.78));
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: 0 18px 38px rgba(15,35,56,0.12);
            padding: 1.15rem;
            backdrop-filter: blur(5px);
        }
        .login-kicker {
            margin: 0;
            font-size: 0.73rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--brand);
        }
        .login-title {
            margin: 0.35rem 0 0.4rem 0;
            font-family: 'Space Grotesk', sans-serif;
            color: var(--ink);
            font-size: clamp(1.22rem, 2vw, 1.55rem);
            line-height: 1.25;
        }
        .login-copy {
            margin: 0;
            font-size: 0.9rem;
            line-height: 1.45;
            color: var(--muted);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left_col, center_col, right_col = st.columns([1.0, 1.1, 1.0])
    with center_col:
        st.markdown(
            """
            <div class="login-card">
                <p class="login-kicker">Secure Access</p>
                <h2 class="login-title">Sign in to CardioInsight</h2>
                <p class="login-copy">Use your authorized email and password to continue.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form", clear_on_submit=False):
            login_email = st.text_input("Email", placeholder="name@example.com")
            login_password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            if _check_credentials(login_email, login_password):
                st.session_state.logged_in = True
                st.session_state.user_email = login_email.strip().lower()
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid email or password")


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
        "bg_1": "#fff4f3",
        "bg_2": "#ffe8e6",
        "ink": "#321512",
        "muted": "#7a4c47",
        "brand": "#c73737",
        "brand_2": "#e04b43",
        "accent": "#ff8a65",
        "surface": "rgba(255, 255, 255, 0.76)",
        "border": "rgba(80, 28, 25, 0.16)",
        "sidebar_grad": "linear-gradient(160deg, #4a1618 0%, #7f2124 45%, #b33a35 100%)",
    },
}

selected_theme = "Sunset Clinic"
theme = THEMES[selected_theme]

st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {{
            --bg1: {theme['bg_1']};
            --bg2: {theme['bg_2']};
            --ink: {theme['ink']};
            --muted: {theme['muted']};
            --brand: {theme['brand']};
            --brand2: {theme['brand_2']};
            --accent: {theme['accent']};
            --surface: {theme['surface']};
            --border: {theme['border']};
            --sidebar-grad: {theme['sidebar_grad']};
        }}

        .stApp {{
            font-family: 'Manrope', sans-serif;
            background: radial-gradient(circle at 20% 10%, rgba(42,157,143,0.12), transparent 36%),
                                    radial-gradient(circle at 90% 0%, rgba(255,127,80,0.10), transparent 32%),
                                    linear-gradient(120deg, var(--bg1) 0%, var(--bg2) 100%);
            color: var(--ink);
        }}

        [data-testid="stSidebar"] {{
            background: var(--sidebar-grad);
            border-right: 1px solid rgba(255,255,255,0.08);
        }}
        [data-testid="stSidebar"] * {{ color: #ecf3ff !important; }}
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] label {{
            border-radius: 12px;
            padding: 0.25rem 0.5rem;
            transition: background 200ms ease;
        }}
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {{
            background: rgba(255,255,255,0.14);
        }}

        .hero-shell {{
            background: linear-gradient(135deg, rgba(255,255,255,0.80), rgba(255,255,255,0.62));
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 1.15rem 1.2rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 18px 34px rgba(15,35,56,0.09);
            backdrop-filter: blur(5px);
            animation: riseIn 600ms ease-out;
        }}
        .hero-kicker {{
            margin: 0;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--brand);
            font-weight: 800;
        }}
        .hero-title {{
            margin: 0.2rem 0 0.35rem 0;
            font-size: clamp(1.45rem, 2.8vw, 2.05rem);
            line-height: 1.2;
            font-family: 'Space Grotesk', sans-serif;
            color: var(--ink);
        }}
        .hero-subtitle {{
            margin: 0;
            color: var(--muted);
            font-size: 0.96rem;
        }}
        .hero-chips {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.8rem;
        }}
        .hero-chip {{
            font-size: 0.76rem;
            font-weight: 700;
            color: #ffffff;
            background: linear-gradient(120deg, var(--brand), var(--brand2));
            border-radius: 999px;
            padding: 0.25rem 0.65rem;
            box-shadow: 0 8px 16px rgba(0,109,119,0.24);
        }}

        .section-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.02rem;
            font-weight: 700;
            margin: 0.6rem 0 0.25rem 0;
            color: var(--ink);
        }}
        .subtle {{
            margin: 0 0 0.4rem 0;
            color: var(--muted);
            font-size: 0.88rem;
            font-weight: 600;
        }}

        .metric-card {{
            position: relative;
            overflow: hidden;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 0.8rem 0.9rem;
            backdrop-filter: blur(3px);
            box-shadow: 0 12px 24px rgba(15,35,56,0.08);
            animation: riseIn 520ms ease-out;
        }}
        .metric-card::after {{
            content: "";
            position: absolute;
            inset: 0;
            transform: translateX(-130%);
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.48), transparent);
            animation: shimmer 2400ms ease-in-out infinite;
            pointer-events: none;
        }}
        .metric-label {{
            margin: 0;
            font-size: 0.8rem;
            font-weight: 700;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.07em;
        }}
        .metric-value {{
            margin: 0.28rem 0 0.35rem 0;
            font-size: clamp(1.25rem, 2.6vw, 1.75rem);
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            color: var(--ink);
            line-height: 1.1;
        }}
        .metric-badge {{
            display: inline-block;
            font-size: 0.74rem;
            font-weight: 700;
            border-radius: 999px;
            padding: 0.2rem 0.58rem;
        }}
        .tone-low {{ background: rgba(42,157,143,0.16); color: #0b6e63; }}
        .tone-moderate {{ background: rgba(244,162,97,0.18); color: #9c5d1f; }}
        .tone-high {{ background: rgba(188,71,73,0.18); color: #8f2f34; }}
        .tone-neutral {{ background: rgba(15,35,56,0.10); color: var(--ink); }}

        .meter-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 0.8rem 0.9rem;
            box-shadow: 0 10px 22px rgba(15,35,56,0.08);
            animation: riseIn 600ms ease-out;
        }}
        .meter-head {{
            display: flex;
            justify-content: space-between;
            font-size: 0.88rem;
            font-weight: 700;
            color: var(--ink);
            margin-bottom: 0.45rem;
        }}
        .meter-track {{
            width: 100%;
            height: 11px;
            border-radius: 999px;
            background: rgba(15,35,56,0.12);
            overflow: hidden;
        }}
        .meter-fill {{
            height: 100%;
            border-radius: 999px;
            animation: fillGrow 900ms ease-out;
            box-shadow: 0 0 0 1px rgba(255,255,255,0.22) inset;
        }}
        .meter-foot {{
            margin-top: 0.4rem;
            font-size: 0.82rem;
            color: var(--muted);
            font-weight: 600;
        }}

        .stButton > button,
        .stFormSubmitButton > button {{
            border: none;
            border-radius: 14px;
            font-weight: 700;
            background: linear-gradient(135deg, var(--brand), var(--brand2));
            color: #ffffff;
            box-shadow: 0 10px 24px rgba(0,109,119,0.28);
            transition: transform 180ms ease, box-shadow 180ms ease;
        }}
        .stButton > button:hover,
        .stFormSubmitButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 14px 28px rgba(0,109,119,0.35);
        }}

        .stNumberInput label,
        .stSelectbox label,
        .stTextInput label,
        .stTextArea label,
        .stDateInput label,
        .stTimeInput label,
        .stMultiSelect label,
        .stRadio label,
        .stCheckbox label,
        .stMarkdown,
        .stCaption,
        .subtle,
        .section-title {{
            color: #111111 !important;
        }}

        [data-baseweb="input"] input,
        [data-baseweb="input"] textarea,
        [data-baseweb="select"] > div,
        [data-testid="stNumberInput"] button,
        [data-testid="stNumberInput"] svg,
        [data-testid="stSelectbox"] svg {{
            color: #111111 !important;
            fill: #111111 !important;
        }}

        [data-baseweb="input"] > div,
        [data-baseweb="select"] > div {{
            background: rgba(255, 255, 255, 0.95) !important;
            border: 1px solid rgba(50, 21, 18, 0.22) !important;
            box-shadow: none !important;
        }}

        @keyframes riseIn {{
            from {{ opacity: 0; transform: translateY(14px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes shimmer {{
            from {{ transform: translateX(-130%); }}
            to {{ transform: translateX(130%); }}
        }}
        @keyframes fillGrow {{
            from {{ width: 0; }}
        }}

        @media (max-width: 900px) {{
            .hero-shell {{ padding: 1rem; }}
            .metric-card {{ padding: 0.72rem 0.78rem; }}
            .meter-card {{ padding: 0.72rem 0.78rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
)

_init_auth_state()
if not st.session_state.logged_in:
    render_login_screen()
    st.stop()

st.sidebar.title("CardioInsight")
st.sidebar.write("Heart disease screening workspace")
st.sidebar.caption(f"Signed in as {st.session_state.user_email}")
if st.sidebar.button("Logout", use_container_width=True):
    st.session_state.logged_in = False
    st.session_state.user_email = ""
    st.rerun()

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
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=250, value=120)
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, value=80)
            height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
            weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=60.0)

        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"], index=0)
            blood_sugar = st.selectbox("Blood Sugar", ["Normal", "Low", "High"], index=0)
            smoking_drinking = st.selectbox("Smoking / Drinking", ["No", "Yes"], index=0)
            glucose = st.selectbox("Glucose", ["Normal", "Low", "High"], index=0)
            cholesterol = st.selectbox("Cholesterol", ["Normal", "Low", "High"], index=0)

        btn_left, btn_center, btn_right = st.columns([1, 1, 1])
        with btn_center:
            submitted = st.form_submit_button("Calculate Risk", use_container_width=True)

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
            "smoking": smoking_drinking,
            "drinking": smoking_drinking,
            "yoga": "No",
            "exercise": "No",
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
            low_count = sum([
                cholesterol == "Low",
                glucose == "Low",
                blood_sugar == "Low",
            ])
            low_penalty = min(0.04, 0.02 * low_count)
            probability = min(1.0, base_probability + low_penalty)
            confidence = probability if prediction == 1 else (1.0 - probability)
            bp_status, bp_note, bp_message_type = classify_blood_pressure(int(systolic_bp), int(diastolic_bp))
            overall_level, combined_probability = combined_risk_level(probability, bp_status)
            bmi, bmi_status = bmi_category(float(height), float(weight))
            progress.empty()

            st.markdown('<p class="section-title">Prediction Summary</p>', unsafe_allow_html=True)
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            with summary_col1:
                render_metric_chip("Heart Probability", f"{probability * 100:.2f}%")
            with summary_col2:
                render_metric_chip("Prediction Confidence", f"{confidence * 100:.2f}%")
            with summary_col3:
                tone = "high" if overall_level == "High" else "moderate" if overall_level == "Moderate" else "low"
                render_metric_chip("Combined Risk", f"{combined_probability * 100:.2f}%", badge_text=overall_level, badge_tone=tone)

            meter_col1, meter_col2 = st.columns(2)
            with meter_col1:
                render_meter(
                    "Risk meter (combined)",
                    combined_probability * 100,
                    f"BP category: {bp_status}",
                    "linear-gradient(90deg, #f7b267 0%, #e76f51 52%, #bc4749 100%)",
                )
            with meter_col2:
                bmi_scaled = min(100.0, max(0.0, (bmi / 40.0) * 100.0))
                render_meter(
                    "BMI meter",
                    bmi_scaled,
                    f"BMI: {bmi:.1f} ({bmi_status})",
                    "linear-gradient(90deg, #ffb4a2 0%, #e76f51 55%, #b23a48 100%)",
                )
