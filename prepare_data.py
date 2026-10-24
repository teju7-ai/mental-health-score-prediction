from pathlib import Path
import pandas as pd

project_folder = Path(__file__).resolve().parent
data_folder = project_folder / "data"

df = pd.read_csv(data_folder / "data.csv", sep=None, engine="python")

# These 14 questions define the DASS-42 stress score.
stress_items = [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]
stress_columns = [f"Q{i}A" for i in stress_items]

# Keep only rows with valid answers for every target-score question.
valid_target = df[stress_columns].isin([1, 2, 3, 4]).all(axis=1)
df = df.loc[valid_target].copy()

# Convert 1–4 responses to 0–3, then sum: possible total is 0–42.
df["stress_score"] = (df[stress_columns] - 1).sum(axis=1)

# Use personality, age, and education as our initial model inputs.
personality_columns = [f"TIPI{i}" for i in range(1, 11)]
feature_columns = personality_columns + ["age", "education"]

prepared = df[feature_columns + ["stress_score"]].copy()

# Invalid personality and education responses become missing values.
for column in personality_columns:
    prepared[column] = prepared[column].where(
        prepared[column].isin(range(1, 8))
    )

prepared["education"] = prepared["education"].where(
    prepared["education"].isin([1, 2, 3, 4])
)

# Practical project cleaning rule, not a clinical age limit:
# treat ages outside 13–100 as missing rather than guessing corrections.
prepared["age"] = pd.to_numeric(prepared["age"], errors="coerce")
prepared["age"] = prepared["age"].where(
    prepared["age"].between(13, 100)
)

# Leave missing inputs for the training pipeline to handle after splitting.
output_path = data_folder / "prepared_data.csv"
prepared.to_csv(output_path, index=False)

print(f"Prepared rows: {len(prepared):,}")
print("\nStress score summary:")
print(prepared["stress_score"].describe())

print("\nMissing values after cleaning:")
print(prepared.isna().sum())

print(f"\nSaved to: {output_path}")