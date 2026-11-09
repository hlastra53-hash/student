# Student Performance Prediction System

A classroom prototype that estimates whether a student is at risk of failing the final grade.

## Problem
Academic risk is often noticed only after poor results. This prototype explores early
identification of students who may need additional academic support using available
student indicators (study time, absences, failures, G1, G2, and others).

## Dataset
UCI Machine Learning Repository - Student Performance (ID 320):
https://archive.ics.uci.edu/dataset/320/student+performance
Cortez, P. (2008). DOI: 10.24432/C5TG7T. License: CC BY 4.0.

## Prediction target
- G3 < 10 = AT RISK
- G3 >= 10 = NOT AT RISK

G3 is the target only and is never used as an input feature.

## Technologies
Python 3.10+, pandas, scikit-learn (Random Forest), joblib, Streamlit, GitHub

## Installation and run
1. Install Python 3.10 or later.
2. Install packages: `pip install -r requirements.txt`
3. Download the data and train the model: `python train_model.py`
   (creates `student_data.csv` and `student_model.pkl`)
4. Run the app: `streamlit run app.py`
5. Open http://localhost:8501 and upload a CSV (a sample can be downloaded in the app).

## Limitations
- Trained on one Portuguese secondary-school dataset; it may not generalize to another
  institution or population.
- Predictions are not diagnoses and must not be used as the sole basis for real student decisions.
