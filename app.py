import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent))

import numpy as np, pandas as pd, joblib, streamlit as st
from src.preprocess import clean_rules, add_features

@st.cache_resource
def load():
    return joblib.load(pathlib.Path(__file__).parent / "models" / "credit_model.joblib")

saved = load()
model, features, threshold = saved["model"], saved["features"], saved["threshold"]

st.title("Credit Risk Scorer")
st.caption("Demo trained on public Kaggle data. Not for real lending decisions.")

c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age", 18, 100, 40)
    income_known = st.checkbox("Monthly income known", value=True)
    income = st.number_input("Monthly income", 0, 1_000_000, 5000, disabled=not income_known)
    dependents = st.number_input("Number of dependents", 0, 20, 0)
    open_lines = st.number_input("Open credit lines and loans", 0, 60, 8)
    real_estate = st.number_input("Real estate loans", 0, 50, 1)
with c2:
    utilization = st.number_input("Credit utilization (1.0 = at the limit)", 0.0, 5.0, 0.3, step=0.05)
    debt_ratio = st.number_input("Debt ratio", 0.0, 10.0, 0.4, step=0.05)
    late_30 = st.number_input("Times 30-59 days late", 0, 20, 0)
    late_60 = st.number_input("Times 60-89 days late", 0, 20, 0)
    late_90 = st.number_input("Times 90+ days late", 0, 20, 0)

if st.button("Score applicant"):
    row = pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines": utilization,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": late_30,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": income if income_known else np.nan,
        "NumberOfOpenCreditLinesAndLoans": open_lines,
        "NumberOfTimes90DaysLate": late_90,
        "NumberRealEstateLoansOrLines": real_estate,
        "NumberOfTime60-89DaysPastDueNotWorse": late_60,
        "NumberOfDependents": dependents,
    }])
    X = add_features(clean_rules(row))[features]    # same prep as training, same column order
    p = model.predict_proba(X)[0, 1]

    st.metric("Estimated default risk", f"{p:.1%}")
    if p >= threshold:
        st.error(f"Flagged: above the {threshold:.0%} cost-based cutoff")
    else:
        st.success(f"Not flagged: below the {threshold:.0%} cost-based cutoff")