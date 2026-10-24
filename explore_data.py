from pathlib import Path
import pandas as pd

# Find the dataset relative to this script.
project_folder = Path(__file__).resolve().parent
data_path = project_folder / "data" / "data.csv"

# Detect the separator: a .csv file can use tabs instead of commas.
df = pd.read_csv(data_path, sep=None, engine="python")

print("Dataset loaded successfully!")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

print("\nFirst 10 column names:")
print(df.columns[:10].tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns with the most missing values:")
print(df.isna().sum().sort_values(ascending=False).head(10))

# Check questionnaire answers against the codebook's valid ranges.
dass_columns = [f"Q{i}A" for i in range(1, 43)]
personality_columns = [f"TIPI{i}" for i in range(1, 11)]

invalid_dass = ~df[dass_columns].isin([1, 2, 3, 4])
invalid_personality = ~df[personality_columns].isin(
    [1, 2, 3, 4, 5, 6, 7]
)

print("\nRows with invalid or missing DASS answers:")
print(invalid_dass.any(axis=1).sum())

print("\nInvalid or missing answers per personality question:")
print(invalid_personality.sum())

print("\nAge summary:")
print(pd.to_numeric(df["age"], errors="coerce").describe())

print("\nEducation values and counts:")
print(df["education"].value_counts(dropna=False).sort_index())

print("\nExact duplicate rows:")
print(df.duplicated().sum())