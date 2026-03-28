import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)


MODEL_PATH = os.path.join("models", "random_forest_model.joblib")
X_TEST_PATH = os.path.join("data", "X_test.csv")
Y_TEST_PATH = os.path.join("data", "y_test.csv")
OUTPUT_DIR = os.path.join("outputs", "evaluation")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)["heart_disease"]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, digits=4)
    report_df = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True)
    ).transpose()

    cm = confusion_matrix(y_test, y_pred)

    # Save confusion matrix heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_heatmap.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # ROC-AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", color="#1f77b4")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(OUTPUT_DIR, "roc_auc_curve.png")
    plt.savefig(roc_path, dpi=200)
    plt.close()

    # Save detailed report to files for reproducibility.
    report_txt_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    report_csv_path = os.path.join(OUTPUT_DIR, "classification_report.csv")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    report_df.to_csv(report_csv_path, index=True)

    print(f"Accuracy Score: {acc:.4f}")
    print("\nClassification Report (Precision, Recall, F1-Score):")
    print(report_text)
    print("Confusion Matrix:")
    print(cm)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Saved confusion matrix heatmap: {cm_path}")
    print(f"Saved ROC-AUC curve: {roc_path}")


if __name__ == "__main__":
    main()
