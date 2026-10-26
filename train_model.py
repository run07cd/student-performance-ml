"""Train and evaluate a beginner-friendly student performance model.

The included CSV is synthetic demo data, not real student records.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student_performance.csv"
MODEL_PATH = ROOT / "models" / "student_score_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

FEATURES = [
    "study_hours_per_day",
    "attendance_percent",
    "previous_score",
    "assignment_completion_percent",
    "sleep_hours",
]
TARGET = "final_score"


def main():
    data = pd.read_csv(DATA_PATH)
    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 3),
        "rmse": round(float(mean_squared_error(y_test, predictions) ** 0.5), 3),
        "r2": round(float(r2_score(y_test, predictions)), 3),
        "test_rows": int(len(y_test)),
        "training_rows": int(len(y_train)),
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print("Model trained successfully.")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
