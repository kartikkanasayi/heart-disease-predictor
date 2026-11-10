import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Heart Disease Classifier", page_icon="❤️", layout="centered")

DATA_FILE = Path(__file__).parent / "heart_disease_dataset_10000.csv"
TARGET = "heart_disease"
NUMERIC_COLUMNS = [
    "age", "resting_blood_pressure_mmHg", "cholesterol_mg_dL", "bmi",
    "fasting_blood_sugar_high", "diabetes", "smoking",
    "family_history_heart_disease", "exercise_induced_angina",
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

def stratified_split(x, y, test_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_fraction)))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]

def train_logistic_regression(x, y, learning_rate=0.03, epochs=3000):
    weights = np.zeros(x.shape[1], dtype=float)
    bias = 0.0
    n = len(y)
    for _ in range(epochs):
        probabilities = sigmoid(x @ weights + bias)
        error = probabilities - y
        dw = (x.T @ error) / n + (0.001 / n) * weights
        db = np.mean(error)
        weights -= learning_rate * dw
        bias -= learning_rate * db
    return weights, bias

@st.cache_resource
def load_and_train():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE.name}")
    data = pd.read_csv(DATA_FILE)
    required = FEATURE_COLUMNS + [TARGET]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Dataset is missing columns: " + ", ".join(missing))
    data = data.dropna(subset=required).copy()
    y = data[TARGET].astype(int).to_numpy()
    x_frame = prepare_features(data)
    feature_names = x_frame.columns.tolist()
    x = x_frame.to_numpy(dtype=float)
    x_train, x_test, y_train, y_test = stratified_split(x, y)
    means = x_train.mean(axis=0)
    stds = x_train.std(axis=0)
    stds[stds == 0] = 1.0
    x_train_scaled = (x_train - means) / stds
    x_test_scaled = (x_test - means) / stds
    weights, bias = train_logistic_regression(x_train_scaled, y_train)
    test_probs = sigmoid(x_test_scaled @ weights + bias)
    test_preds = (test_probs >= 0.5).astype(int)
    accuracy = float(np.mean(test_preds == y_test))
    return feature_names, means, stds, weights, bias, accuracy, len(data)

st.title("❤️ Heart Disease Classifier")
st.write("Enter the patient's details to see the output of the educational machine-learning model.")

try:
    feature_names, means, stds, weights, bias, accuracy, rows = load_and_train()
except Exception as exc:
    st.error(f"Could not load/train the model: {exc}")
    st.info("Keep heart_disease_dataset_10000.csv in the same GitHub repository folder as streamlit_app.py.")
    st.stop()

with st.expander("Model information", expanded=False):
    st.write(f"Training dataset records: **{rows:,}**")
    st.write(f"Hold-out test accuracy: **{accuracy * 100:.1f}%**")
    st.caption("Accuracy is measured on a synthetic dataset and is not evidence of clinical performance.")

with st.form("patient_form"):
    st.subheader("Patient details")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", min_value=6, max_value=80, value=40, step=1)
        sex = st.selectbox("Sex", ["Female", "Male"])
        bp = st.number_input("Resting blood pressure (mmHg)", min_value=60, max_value=250, value=120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=80, max_value=600, value=200)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=24.0, step=0.1)
        max_hr = st.number_input("Maximum heart rate (bpm)", min_value=40, max_value=250, value=160)
    with c2:
        fasting = st.selectbox("High fasting blood sugar?", ["No", "Yes"])
        diabetes = st.selectbox("Diabetes?", ["No", "Yes"])
        smoking = st.selectbox("Currently smoking?", ["No", "Yes"])
        family = st.selectbox("Family history of heart disease?", ["No", "Yes"])
        angina = st.selectbox("Exercise-induced angina?", ["No", "Yes"])
        chest_pain = st.selectbox("Chest pain type", ["Typical", "Atypical", "Non-anginal", "Asymptomatic"])
    submitted = st.form_submit_button("Calculate model estimate", type="primary", use_container_width=True)

if submitted:
    patient = pd.DataFrame([{
        "age": age, "sex": sex, "resting_blood_pressure_mmHg": bp,
        "cholesterol_mg_dL": cholesterol, "bmi": bmi,
        "fasting_blood_sugar_high": int(fasting == "Yes"),
        "diabetes": int(diabetes == "Yes"), "smoking": int(smoking == "Yes"),
        "family_history_heart_disease": int(family == "Yes"),
        "exercise_induced_angina": int(angina == "Yes"),
        "chest_pain_type": chest_pain, "maximum_heart_rate": max_hr,
    }])
    patient_x = prepare_features(patient, feature_names).to_numpy(dtype=float)
    patient_scaled = (patient_x - means) / stds
    probability = float(sigmoid(patient_scaled @ weights + bias)[0])
    percent = probability * 100
    st.subheader("Model output")
    st.metric("Estimated probability from this model", f"{percent:.1f}%")
    st.progress(min(max(probability, 0.0), 1.0))
    st.caption("This percentage is a model output, not a clinically validated estimate of an individual's actual risk.")

st.warning(
    "Educational project only. This model and dataset are not clinically validated and must not be used "
    "to diagnose, screen, or make treatment decisions. For symptoms or health concerns, consult a qualified clinician."
)
