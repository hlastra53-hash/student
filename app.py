# app.py
import json
import os
import subprocess
import sys
 
import joblib
import pandas as pd
import streamlit as st
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "student_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "student_data.csv")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
 
st.set_page_config(page_title="Student Predict", page_icon="🎓", layout="wide")
 
# If the model, dataset or metrics are missing (e.g., fresh clone or Streamlit
# Cloud), run train_model.py once to create them.
if not all(os.path.exists(p) for p in (MODEL_PATH, DATA_PATH, METRICS_PATH)):
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
 
 
@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)
 
 
def feature_importance(model):
    """Random Forest importances, with one-hot columns summed back to the original feature."""
    names = model.named_steps["preprocessor"].get_feature_names_out()
    values = model.named_steps["classifier"].feature_importances_
    originals = list(model.feature_names_in_)
    totals = {}
    for name, value in zip(names, values):
        clean = name.split("__", 1)[1]
        if name.startswith("categorical__"):
            matches = [c for c in originals if clean.startswith(c + "_")]
            clean = max(matches, key=len) if matches else clean
        totals[clean] = totals.get(clean, 0.0) + float(value)
    return pd.Series(totals).sort_values(ascending=False)
 
 
model = load_model()
metrics = load_metrics()
data = pd.read_csv(DATA_PATH)
data["Risk status"] = data["G3"].apply(lambda g: "AT RISK" if g < 10 else "NOT AT RISK")
 
st.title("🎓 Student Performance Prediction")
st.caption("Classroom prototype for academic risk exploration")
st.info(
    "Responsible use: this is a classroom demonstration using public, historical data. "
    "Predictions are not diagnoses and must not be used as the sole basis for real student decisions."
)
 
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", len(data))
c2.metric("Average final grade", round(data["G3"].mean(), 2))
c3.metric("Historical at-risk records", int((data["G3"] < 10).sum()))
c4.metric("Historical at-risk rate", f"{(data['G3'] < 10).mean() * 100:.1f}%")
 
tab_overview, tab_model, tab_predict = st.tabs(
    ["📊 Data overview", "🧪 Model evaluation", "🔮 Predict"]
)
 
# ---------------------------------------------------------------- Overview
with tab_overview:
    st.subheader("Dataset preview")
    st.dataframe(data.drop(columns=["Risk status"]).head(10), width="stretch")
 
    left, right = st.columns(2)
    with left:
        st.subheader("Final grade (G3) distribution")
        st.caption("Students scoring below 10 are labelled AT RISK.")
        st.bar_chart(data["G3"].value_counts().sort_index())
    with right:
        st.subheader("At risk vs not at risk")
        st.caption("Historical records, based on G3.")
        st.bar_chart(data["Risk status"].value_counts())
 
    left, right = st.columns(2)
    with left:
        st.subheader("Average final grade by past failures")
        st.bar_chart(data.groupby("failures")["G3"].mean())
    with right:
        st.subheader("Average final grade by weekly study time")
        st.caption("1 = <2 hours, 2 = 2-5 hours, 3 = 5-10 hours, 4 = >10 hours")
        st.bar_chart(data.groupby("studytime")["G3"].mean())
 
    st.subheader("Second-period grade (G2) vs final grade (G3)")
    st.scatter_chart(data, x="G2", y="G3", color="Risk status")
 
# ------------------------------------------------------------- Evaluation
with tab_model:
    report = metrics["report"]
    st.subheader("Model evaluation summary")
    st.caption(
        f"Random Forest, evaluated on a held-out test set of {metrics['test_size']} records "
        f"(trained on {metrics['train_size']})."
    )
 
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    m2.metric("Precision (At risk)", f"{report['At risk']['precision']:.3f}")
    m3.metric("Recall (At risk)", f"{report['At risk']['recall']:.3f}")
    m4.metric("F1-score (At risk)", f"{report['At risk']['f1-score']:.3f}")
    st.caption("Do not interpret accuracy alone. Recall for 'At risk' shows how many truly at-risk students were caught.")
 
    left, right = st.columns(2)
    with left:
        st.subheader("Classification report")
        table = pd.DataFrame(report).T.drop(index=["accuracy"], errors="ignore")
        table = table[["precision", "recall", "f1-score", "support"]].round(3)
        st.dataframe(table, width="stretch")
    with right:
        st.subheader("Confusion matrix")
        cm = pd.DataFrame(
            metrics["confusion_matrix"],
            index=["Actual: Not at risk", "Actual: At risk"],
            columns=["Predicted: Not at risk", "Predicted: At risk"],
        )
        st.dataframe(cm, width="stretch")
        st.caption("Bottom-left cell = at-risk students the model missed.")
 
    st.subheader("Top 10 most important features")
    importance = feature_importance(model).head(10)
    st.bar_chart(importance, horizontal=True)
    st.caption("G1 and G2 are earlier grades, so they strongly predict the final grade.")
 
# ---------------------------------------------------------------- Predict
with tab_predict:
    st.subheader("Predict uploaded student records")
    st.write("Upload a CSV containing the same predictor columns as the training data.")
 
    template = data.drop(columns=["G3", "Risk status"]).head(5)
    st.download_button(
        "Download sample input CSV",
        template.to_csv(index=False),
        "sample_prediction_input.csv",
        "text/csv"
    )
 
    uploaded = st.file_uploader("Choose student CSV", type=["csv"])
 
    if uploaded is not None:
        try:
            new_data = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the file as a CSV: {exc}")
            st.stop()
 
        required_columns = list(model.feature_names_in_)
        missing = [c for c in required_columns if c not in new_data.columns]
 
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            try:
                predictions = model.predict(new_data[required_columns])
            except Exception as exc:
                st.error(f"Could not make predictions. Check the column values: {exc}")
                st.stop()
 
            results = new_data.copy()
            results["Prediction"] = [
                "AT RISK" if value == 1 else "NOT AT RISK"
                for value in predictions
            ]
 
            st.subheader("Prediction results")
            r1, r2, r3 = st.columns(3)
            r1.metric("Records uploaded", len(results))
            r2.metric("Predicted at-risk records", int((predictions == 1).sum()))
            r3.metric("Predicted at-risk rate", f"{(predictions == 1).mean() * 100:.1f}%")
 
            st.dataframe(results, width="stretch")
            st.bar_chart(results["Prediction"].value_counts())
 
            st.download_button(
                "Download prediction results",
                results.to_csv(index=False),
                "student_predictions.csv",
                "text/csv"
            )
 
