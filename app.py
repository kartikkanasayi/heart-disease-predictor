from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from pathlib import Path

app = Flask(__name__)

DATA_FILE = Path(__file__).with_name("heart_disease_dataset_10000.csv")
TARGET = "heart_disease"

NUMERIC_COLUMNS = [
    "age",
    "resting_blood_pressure_mmHg",
    "cholesterol_mg_dL",
    "bmi",
    "fasting_blood_sugar_high",
    "diabetes",
    "smoking",
    "family_history_heart_disease",
    "exercise_induced_angina",
    "maximum_heart_rate",
]
CATEGORICAL_COLUMNS = ["sex", "chest_pain_type"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

def sigmoid(z):
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))

def prepare_features(frame, expected_columns=None):
    x = frame[FEATURE_COLUMNS].copy()
    x["sex"] = x["sex"].astype(str).str.strip().str.title()
    x["chest_pain_type"] = x["chest_pain_type"].astype(str).str.strip().str.title()
    x = pd.get_dummies(x, columns=CATEGORICAL_COLUMNS, dtype=float)
    if expected_columns is not None:
        x = x.reindex(columns=expected_columns, fill_value=0)
    return x.astype(float)

def train_logistic_regression(x, y, learning_rate=0.08, epochs=4000):
    weights = np.zeros(x.shape[1], dtype=float)
    bias = 0.0
    n = len(y)
    for _ in range(epochs):
        probabilities = sigmoid(x @ weights + bias)
        error = probabilities - y
        dw = (x.T @ error) / n
        db = np.mean(error)
        dw += (0.001 / n) * weights
        weights -= learning_rate * dw
        bias -= learning_rate * db
    return weights, bias

def load_and_train():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found. Put '{DATA_FILE.name}' in the same folder as app.py."
        )
    data = pd.read_csv(DATA_FILE)
    required = FEATURE_COLUMNS + [TARGET]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))
    data = data.dropna(subset=required).copy()
    if data.empty:
        raise ValueError("The CSV has no complete patient records.")
    y = data[TARGET].astype(int).to_numpy()
    if not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) < 2:
        raise ValueError("The heart_disease column must contain both 0 and 1.")
    x_frame = prepare_features(data)
    feature_names = x_frame.columns.tolist()
    x = x_frame.to_numpy(dtype=float)

    # Stratified 80/20 split, matching the supplied command-line model.
    rng = np.random.default_rng(42)
    train_idx, test_idx = [], []
    for class_value in np.unique(y):
        class_idx = np.where(y == class_value)[0]
        rng.shuffle(class_idx)
        test_count = max(1, int(round(len(class_idx) * 0.20)))
        test_idx.extend(class_idx[:test_count])
        train_idx.extend(class_idx[test_count:])
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    means = x_train.mean(axis=0)
    stds = x_train.std(axis=0)
    stds[stds == 0] = 1.0
    x_train_scaled = (x_train - means) / stds
    weights, bias = train_logistic_regression(x_train_scaled, y_train)

    test_probs = sigmoid(((x[test_idx] - means) / stds) @ weights + bias)
    test_preds = (test_probs >= 0.5).astype(int)
    accuracy = float(np.mean(test_preds == y_test))
    return {
        "weights": weights, "bias": bias, "means": means, "stds": stds,
        "feature_names": feature_names, "accuracy": accuracy,
        "rows_used": len(data)
    }

MODEL = None
MODEL_ERROR = None
try:
    MODEL = load_and_train()
except Exception as exc:
    MODEL_ERROR = str(exc)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = MODEL_ERROR
    if request.method == "POST" and MODEL is not None:
        try:
            patient = {
                "age": int(request.form["age"]),
                "sex": request.form["sex"],
                "resting_blood_pressure_mmHg": int(request.form["blood_pressure"]),
                "cholesterol_mg_dL": int(request.form["cholesterol"]),
                "bmi": float(request.form["bmi"]),
                "fasting_blood_sugar_high": int(request.form["fasting_sugar"]),
                "diabetes": int(request.form["diabetes"]),
                "smoking": int(request.form["smoking"]),
                "family_history_heart_disease": int(request.form["family_history"]),
                "exercise_induced_angina": int(request.form["angina"]),
                "chest_pain_type": request.form["chest_pain_type"],
                "maximum_heart_rate": int(request.form["max_heart_rate"]),
            }
            if not 6 <= patient["age"] <= 80:
                raise ValueError("Age must be between 6 and 80.")
            if not 60 <= patient["resting_blood_pressure_mmHg"] <= 250:
                raise ValueError("Blood pressure must be between 60 and 250 mmHg.")
            if not 80 <= patient["cholesterol_mg_dL"] <= 600:
                raise ValueError("Cholesterol must be between 80 and 600 mg/dL.")
            if not 10 <= patient["bmi"] <= 70:
                raise ValueError("BMI must be between 10 and 70.")
            if not 40 <= patient["maximum_heart_rate"] <= 250:
                raise ValueError("Maximum heart rate must be between 40 and 250 bpm.")
            for key in ["fasting_blood_sugar_high", "diabetes", "smoking",
                        "family_history_heart_disease", "exercise_induced_angina"]:
                if patient[key] not in (0, 1):
                    raise ValueError("Yes/No fields must be selected correctly.")

            patient_frame = pd.DataFrame([patient])
            features = prepare_features(
                patient_frame, MODEL["feature_names"]
            ).to_numpy(dtype=float)
            scaled = (features - MODEL["means"]) / MODEL["stds"]
            probability = float(sigmoid(scaled @ MODEL["weights"] + MODEL["bias"])[0])
            result = {
                "percent": probability * 100,
                "accuracy": MODEL["accuracy"] * 100
            }
        except (ValueError, KeyError) as exc:
            error = str(exc)

    return render_template(
        "index.html", result=result, error=error,
        model_ready=MODEL is not None,
        accuracy=(MODEL["accuracy"] * 100 if MODEL else None),
        rows_used=(MODEL["rows_used"] if MODEL else None)
    )

if __name__ == "__main__":
    app.run(debug=True)
