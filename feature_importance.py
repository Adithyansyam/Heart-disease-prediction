import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MODEL_PATH = os.path.join("models", "random_forest_model.joblib")
X_TRAIN_PATH = os.path.join("data", "X_train.csv")
OUTPUT_DIR = os.path.join("outputs", "feature_importance")
PLOT_PATH = os.path.join(OUTPUT_DIR, "feature_importance_horizontal.png")
CSV_PATH = os.path.join(OUTPUT_DIR, "feature_importance_sorted.csv")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    X_train = pd.read_csv(X_TRAIN_PATH)

    feature_importances = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    feature_importances.to_csv(CSV_PATH, index=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))
    plt.barh(feature_importances["feature"], feature_importances["importance"], color="#4c78a8")
    plt.gca().invert_yaxis()
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance (Most to Least)")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()

    print("Feature importance plot created successfully.")
    print(f"Saved plot: {PLOT_PATH}")
    print(f"Saved sorted importances: {CSV_PATH}")
    print("Top 5 features:")
    for idx, row in feature_importances.head(5).iterrows():
        print(f"- {row['feature']}: {row['importance']:.6f}")


if __name__ == "__main__":
    main()
