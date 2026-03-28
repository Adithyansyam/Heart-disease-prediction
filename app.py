import os
from io import BytesIO

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.title("Heart Disease Prediction using Random Forest")
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
st.dataframe(df.head(), use_container_width=True)
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
    st.dataframe(report_df, use_container_width=True)

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
