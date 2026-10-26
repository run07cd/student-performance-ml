"""Streamlit interface for the student performance demo model."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "student_score_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")
st.title("🎓 Student Performance Predictor")
st.write("A beginner machine-learning project that estimates a demo score from study habits and academic indicators.")

st.info(
    "Educational demo only: the included dataset is synthetic. This model is not validated "
    "for real student assessment and should not be used to make educational decisions."
)

if not MODEL_PATH.exists():
    st.warning("The model has not been trained yet. Run python train_model.py first.")
    st.stop()

with st.form("prediction_form"):
    st.subheader("Enter sample information")
    study_hours = st.slider("Study hours per day", 0.0, 12.0, 3.0, 0.5)
    attendance = st.slider("Attendance (%)", 0.0, 100.0, 80.0, 1.0)
    previous_score = st.slider("Previous score (%)", 0.0, 100.0, 65.0, 1.0)
    assignments = st.slider("Assignment completion (%)", 0.0, 100.0, 75.0, 1.0)
    sleep_hours = st.slider("Sleep hours per night", 0.0, 12.0, 7.0, 0.5)
    submitted = st.form_submit_button("Predict demo score")

if submitted:
    sample = pd.DataFrame([{
        "study_hours_per_day": study_hours,
        "attendance_percent": attendance,
        "previous_score": previous_score,
        "assignment_completion_percent": assignments,
        "sleep_hours": sleep_hours,
    }])
    prediction = float(joblib.load(MODEL_PATH).predict(sample)[0])
    bounded_prediction = max(0, min(100, prediction))
    st.metric("Estimated demo score", f"{bounded_prediction:.1f} / 100")
    st.caption("This estimate comes from synthetic training data and is not a reliable prediction of a real student's marks.")

if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text())
    st.subheader("Model evaluation")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", metrics["mae"])
    c2.metric("RMSE", metrics["rmse"])
    c3.metric("R²", metrics["r2"])
    st.caption(f"Evaluated on {metrics['test_rows']} held-out synthetic examples. Metrics are illustrative.")

st.subheader("What this project demonstrates")
st.markdown(
    "- Loading and exploring tabular data with Pandas\n"
    "- Splitting data into training and test sets\n"
    "- Building a Scikit-learn preprocessing/model pipeline\n"
    "- Evaluating regression with MAE, RMSE, and R²\n"
    "- Serving predictions through a Streamlit interface"
)
