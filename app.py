# app.py
import os
import subprocess
import sys

import streamlit as st
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "student_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "student_data.csv")

st.set_page_config(page_title="Student Predict", page_icon="🎓", layout="wide")

# If the model or dataset is missing (e.g., fresh clone or Streamlit Cloud),
# run train_model.py once to create student_data.csv and student_model.pkl.
if not (os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH)):
    with st.spinner("First run: downloading the dataset and training the model..."):
        result = subprocess.run(
            [sys.executable, "train_model.py"],
            cwd=BASE_DIR, capture_output=True, text=True
        )
    if result.returncode != 0 or not os.path.exists(MODEL_PATH):
        st.error("Could not create the model. Run `python train_model.py` first.")
        st.code(result.stderr or result.stdout)
        st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
data = pd.read_csv(DATA_PATH)

st.title("🎓 Student Performance Prediction")
st.caption("Classroom prototype for academic risk exploration")

c1, c2, c3 = st.columns(3)
c1.metric("Records", len(data))
c2.metric("Average final grade", round(data["G3"].mean(), 2))
c3.metric("Historical at-risk records", int((data["G3"] < 10).sum()))

st.subheader("Dataset preview")
st.dataframe(data.head(10), width="stretch")

st.subheader("Predict uploaded student records")
st.write("Upload a CSV containing the same predictor columns as the training data.")

template = data.drop(columns=["G3"]).head(5)
st.download_button(
    "Download sample input CSV",
    template.to_csv(index=False),
    "sample_prediction_input.csv",
    "text/csv"
)

uploaded = st.file_uploader("Choose student CSV", type=["csv"])

if uploaded is not None:
    new_data = pd.read_csv(uploaded)
    required_columns = list(model.feature_names_in_)
    missing = [c for c in required_columns if c not in new_data.columns]

    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
    else:
        predictions = model.predict(new_data[required_columns])
        results = new_data.copy()
        results["Prediction"] = [
            "AT RISK" if value == 1 else "NOT AT RISK"
            for value in predictions
        ]
        st.subheader("Prediction results")
        st.dataframe(results, width="stretch")
        st.metric("Predicted at-risk records", int((predictions == 1).sum()))
        st.download_button(
            "Download prediction results",
            results.to_csv(index=False),
            "student_predictions.csv",
            "text/csv"
        )
