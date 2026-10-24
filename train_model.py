from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def calculate_metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(root_mean_squared_error(actual, predicted)),
        "R2": float(r2_score(actual, predicted)),
    }


def print_metrics(metrics):
    print(f"MAE:  {metrics['MAE']:.3f}")
    print(f"RMSE: {metrics['RMSE']:.3f}")
    print(f"R²:   {metrics['R2']:.3f}")


def main():
    project_folder = Path(__file__).resolve().parent
    data_path = project_folder / "data" / "prepared_data.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            "prepared_data.csv was not found. Run prepare_data.py first."
        )

    df = pd.read_csv(data_path)

    numeric_features = [f"TIPI{i}" for i in range(1, 11)] + ["age"]
    categorical_features = ["education"]
    feature_columns = numeric_features + categorical_features

    X = df[feature_columns]
    y = df["stress_score"]

    # Reserve 20% for final testing.
    X_development, X_test, y_development, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Overall split: 60% training, 20% validation, 20% testing.
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        random_state=42,
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    ridge_model = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("regressor", Ridge(alpha=1.0)),
    ])

    forest_model = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=10,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Ridge regression": ridge_model,
        "Random forest": forest_model,
    }

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_validation):,}")
    print(f"Reserved test rows: {len(X_test):,}")

    validation_results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...", flush=True)
        model.fit(X_train, y_train)
        predictions = model.predict(X_validation)

        metrics = calculate_metrics(y_validation, predictions)
        validation_results[name] = metrics

        print(f"{name} — validation results")
        print_metrics(metrics)

    # Random Forest was selected using the validation results.
    # The test set must not be used to choose or tune the model.
    print("\nTraining final Random Forest...", flush=True)
    final_model = clone(forest_model)
    final_model.fit(X_development, y_development)

    test_predictions = final_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, test_predictions)

    print("\nFinal Random Forest — test results")
    print_metrics(test_metrics)

    # Save preprocessing and the regressor together.
    models_folder = project_folder / "models"
    models_folder.mkdir(exist_ok=True)

    model_path = models_folder / "stress_model.joblib"
    joblib.dump(final_model, model_path)

    metadata = {
        "model": "RandomForestRegressor",
        "target": "DASS-42 stress score",
        "target_range": [0, 42],
        "features": feature_columns,
        "development_rows": len(X_development),
        "test_rows": len(X_test),
        "random_state": 42,
        "validation_metrics": validation_results,
        "test_metrics": test_metrics,
        "intended_use": "Educational prototype, not clinical diagnosis",
    }

    metadata_path = models_folder / "model_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"\nModel saved: {model_path}")
    print(f"Metadata saved: {metadata_path}")


if __name__ == "__main__":
    main()