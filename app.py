# Streamlit ML Web App Demo: Bucknell Lending Predictor

import streamlit as st
import pandas as pd
import pickle

# -------------------------
# Open the file and load the deployment bundle
# -------------------------
file_to_load = 'bucknell_lending_deployment_bundle.pkl'
with open(file_to_load, 'rb') as file:
    loaded_bundle = pickle.load(file)

classifier_model = loaded_bundle['classifier_model']
regressor_model = loaded_bundle['regressor_model']
scaler = loaded_bundle['scaler']
final_model_columns = loaded_bundle['final_model_columns']
numeric_features = loaded_bundle['numeric_features']
categorical_features = loaded_bundle['categorical_features']
predictor_cols = loaded_bundle['predictor_cols']
dti_median = loaded_bundle['dti_median']

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="🏦 Bucknell Lending Predictor", layout="centered")
st.title("🏦 Bucknell Lending Decision App")
st.markdown("Enter a potential borrower's application details to estimate loan performance.")

# -------------------------
# User input
# -------------------------
loan_amnt = st.slider("Loan Amount", 1000, 40000, 10000, step=500)
term_num = st.selectbox("Loan Term (Months)", [36, 60])
int_rate = st.slider("Interest Rate", 5.3, 30.9, 12.0, step=0.1)

grade = st.selectbox("Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
emp_length = st.selectbox(
    "Employment Length",
    ['Unknown', '< 1 year', '1 year', '2 years', '3 years', '4 years',
     '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
)
home_ownership = st.selectbox("Home Ownership", ['MORTGAGE', 'RENT', 'OWN', 'OTHER', 'NONE', 'ANY'])
annual_inc = st.slider("Annual Income", 100, 6100000, 75000, step=1000)
verification_status = st.selectbox("Verification Status", ['Verified', 'Source Verified', 'Not Verified'])
purpose = st.selectbox(
    "Loan Purpose",
    ['debt_consolidation', 'credit_card', 'home_improvement', 'other',
     'major_purchase', 'medical', 'small_business', 'car', 'vacation',
     'moving', 'house', 'wedding', 'renewable_energy', 'educational']
)

dti = st.slider("Debt-to-Income Ratio", 0.0, 50.0, 15.0, step=0.1)
delinq_2yrs = st.slider("Delinquencies in Last 2 Years", 0, 26, 0)
open_acc = st.slider("Open Accounts", 1, 68, 10)
pub_rec = st.slider("Public Records", 0, 21, 0)
revol_bal = st.slider("Revolving Balance", 0, 959754, 12000, step=500)
revol_util = st.slider("Revolving Utilization", 0.0, 152.0, 50.0, step=0.5)

fico_range_low = st.slider("FICO Range Low", 660, 845, 690)
fico_range_high = st.slider("FICO Range High", 664, 850, 694)
credit_age_years = st.slider("Credit Age (Years)", 0.0, 65.0, 15.0, step=0.5)

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Loan Outcome"):
    fico_avg = (fico_range_low + fico_range_high) / 2

    new_applicant = pd.DataFrame({
        'loan_amnt': [loan_amnt],
        'int_rate': [int_rate],
        'grade': [grade],
        'emp_length': [emp_length],
        'home_ownership': [home_ownership],
        'annual_inc': [annual_inc],
        'verification_status': [verification_status],
        'purpose': [purpose],
        'dti': [dti],
        'delinq_2yrs': [delinq_2yrs],
        'open_acc': [open_acc],
        'pub_rec': [pub_rec],
        'revol_bal': [revol_bal],
        'revol_util': [revol_util],
        'term_num': [term_num],
        'fico_avg': [fico_avg],
        'credit_age_years': [credit_age_years]
    })

    # fill missing dti if needed
    new_applicant['dti'] = new_applicant['dti'].fillna(dti_median)

    # make dummies
    new_applicant = pd.get_dummies(new_applicant, columns=categorical_features, drop_first=True)

    # line up columns with training data
    new_applicant = new_applicant.reindex(columns=final_model_columns, fill_value=0)

    # scale numeric columns
    new_applicant[numeric_features] = scaler.transform(new_applicant[numeric_features])

    # predictions
    prob_fully_paid = classifier_model.predict_proba(new_applicant)[:, 1][0]
    prob_charged_off = 1 - prob_fully_paid
    pred_ret_pess = regressor_model.predict(new_applicant)[0]

    # recommendation rule
    if (prob_fully_paid >= 0.85) and (pred_ret_pess >= 3):
        recommendation = "Approve"
    elif (prob_fully_paid >= 0.75) and (pred_ret_pess >= 0):
        recommendation = "Review Manually"
    else:
        recommendation = "Reject"

    # display results
    st.write(f"Predicted Probability of Fully Paid: **{prob_fully_paid:.4f}**")
    st.write(f"Predicted Probability of Charged Off: **{prob_charged_off:.4f}**")
    st.write(f"Predicted ret_PESS: **{pred_ret_pess:.4f}**")
    st.success(f"Recommended Action: **{recommendation}**")

    chart_data = pd.DataFrame(
        {'Probability': [prob_fully_paid, prob_charged_off]},
        index=['Fully Paid', 'Charged Off']
    )
    st.write("Prediction Breakdown:")
    st.bar_chart(chart_data)

st.markdown("---")
st.markdown("**Bucknell Lending Final Project Demo** | Powered by Streamlit")
