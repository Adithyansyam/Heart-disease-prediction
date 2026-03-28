import os
import numpy as np
import pandas as pd


N_ROWS = 1000
RANDOM_SEED = 42


rng = np.random.default_rng(RANDOM_SEED)

age = rng.integers(25, 81, N_ROWS)
gender = rng.choice(["Male", "Female"], size=N_ROWS, p=[0.52, 0.48])
height = rng.normal(168, 10, N_ROWS).clip(145, 200).round(1)
weight = rng.normal(72, 14, N_ROWS).clip(40, 140).round(1)

cholesterol = rng.choice(["Low", "Medium", "High"], size=N_ROWS, p=[0.3, 0.45, 0.25])
glucose = rng.choice(["Low", "Medium", "High"], size=N_ROWS, p=[0.35, 0.45, 0.2])
blood_sugar = rng.choice(["Low", "Medium", "High"], size=N_ROWS, p=[0.4, 0.4, 0.2])

smoking = rng.choice(["Yes", "No"], size=N_ROWS, p=[0.3, 0.7])
drinking = rng.choice(["Yes", "No"], size=N_ROWS, p=[0.35, 0.65])
yoga = rng.choice(["Yes", "No"], size=N_ROWS, p=[0.35, 0.65])
exercise = rng.choice(["Yes", "No"], size=N_ROWS, p=[0.45, 0.55])
gym = rng.choice(["Yes", "No"], size=N_ROWS, p=[0.3, 0.7])

# Build a risk score so target is not random noise.
risk = (
    0.04 * (age - 45)
    + 0.015 * (weight - 75)
    + 0.7 * (cholesterol == "High")
    + 0.35 * (cholesterol == "Medium")
    + 0.55 * (glucose == "High")
    + 0.25 * (glucose == "Medium")
    + 0.65 * (blood_sugar == "High")
    + 0.3 * (blood_sugar == "Medium")
    + 0.45 * (smoking == "Yes")
    + 0.25 * (drinking == "Yes")
    - 0.22 * (yoga == "Yes")
    - 0.3 * (exercise == "Yes")
    - 0.2 * (gym == "Yes")
)

noise = rng.normal(0, 0.6, N_ROWS)
logit = -1.2 + risk + noise
prob = 1 / (1 + np.exp(-logit))
heart_disease = rng.binomial(1, prob, N_ROWS)

df = pd.DataFrame(
    {
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
        "heart_disease": heart_disease,
    }
)

os.makedirs("data", exist_ok=True)
out_path = os.path.join("data", "heart_disease_synthetic.csv")
df.to_csv(out_path, index=False)

print(f"Dataset generated: {out_path}")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
print("Class distribution:")
print(df["heart_disease"].value_counts().sort_index())
