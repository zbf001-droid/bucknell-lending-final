import streamlit as st
import pandas as pd
import numpy as np
import pickle
import altair as alt

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Bucknell Lending Club",
    page_icon="🦬",
    layout="centered"
)

# -------------------------
# Bucknell Styling
# -------------------------
st.markdown("""
    <style>
    :root {
        --bucknell-blue: #003865;
        --bucknell-orange: #E87722;
        --light-bg: #f9f7f4;
        --light-gray: #e6e8eb;
        --text-dark: #252733;
    }

    .main {
        background-color: var(--light-bg);
    }

    .stApp {
        background-color: var(--light-bg);
    }

    h1 {
        color: var(--bucknell-blue);
    }

    h2 {
        color: var(--bucknell-blue);
    }

    h3 {
        color: var(--bucknell-blue);
    }

    /* Slider label text */
    [data-testid="stSlider"] label {
        color: var(--text-dark) !important;
        font-weight: 500;
    }

    /* Keep slider container clean */
    [data-testid="stSlider"] {
        background-color: transparent !important;
    }

    [data-testid="stSlider"] div {
        box-shadow: none !important;
    }

    /* Hide min/max slider numbers */
    [data-testid="stTickBar"] {
        display: none !important;
    }

    /* Remove any background box around slider numbers */
    [data-testid="stSlider"] span {
        background-color: transparent !important;
    }

    /* Current slider value */
    [data-testid="stSlider"] div[role="slider"] div {
        color: var(--bucknell-blue) !important;
        background-color: transparent !important;
        font-weight: 700 !important;
    }

    /* Slider dot */
    [data-testid="stSlider"] div[role="slider"] {
        background-color: var(--bucknell-orange) !important;
        border-color: var(--bucknell-orange) !important;
        box-shadow: none !important;
    }

    /* Filled slider line */
    [data-testid="stSlider"] div[data-baseweb="slider"] div[style*="background"] {
        background-color: var(--bucknell-orange) !important;
    }

    .stButton>button {
        background-color: var(--bucknell-blue);
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6em 2em;
        border: none;
        width: 100%;
    }

    .stButton>button:hover {
        background-color: var(--bucknell-orange);
        color: white;
    }

    .result-box {
        background-color: var(--bucknell-blue);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }

    .fund {
        background-color: #2e7d32;
    }

    .review {
        background-color: var(--bucknell-orange);
    }

    .decline {
        background-color: #b71c1c;
    }

    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid var(--bucknell-orange);
        margin: 8px 0;
    }

    .footer {
        text-align: center;
        color: gray;
        font-size: 12px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Load Models
# -------------------------
with open('bucknell_lending_deployment_bundle.pkl', 'rb') as f:
    bundle = pickle.load(f)

classifier = bundle['classifier']
regressor = bundle['regressor']
scaler = bundle['scaler']
feature_columns = bundle['feature_columns']

# -------------------------
# Header
# -------------------------
st.markdown("<h1 style='text-align:center;'>🦬 Bucknell Lending Club</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; color:#E87722;'>Loan Decision Support System</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Helping Bucknell Lending Club make smarter funding decisions, one application at a time.</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------
# Input Form
# -------------------------
st.markdown("### 📋 Loan Details")
col1, col2 = st.columns(2)

with col1:
    loan_amnt = st.slider("Loan Amount ($)", 1000, 40000, 10000, step=500)
    term_num = st.selectbox("Loan Term (Months)", [36, 60])
    int_rate = st.slider("Interest Rate (%)", 5.3, 30.9, 12.0, step=0.1)
    grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    purpose = st.selectbox("Loan Purpose", [
        'debt_consolidation', 'credit_card', 'home_improvement', 'other',
        'major_purchase', 'medical', 'small_business', 'car', 'vacation',
        'moving', 'house', 'wedding', 'renewable_energy', 'educational'
    ])

with col2:
    annual_inc = st.slider("Annual Income ($)", 10000, 500000, 75000, step=1000)
    dti = st.slider("Debt-to-Income Ratio", 0.0, 40.0, 15.0, step=0.1)
    revol_util = st.slider("Revolving Utilization (%)", 0.0, 100.0, 50.0, step=0.5)
    revol_bal = st.slider("Revolving Balance ($)", 0, 200000, 12000, step=500)
    fico_score = st.slider("FICO Score", 660, 850, 692)

st.markdown("### 👤 Borrower Profile")
col3, col4 = st.columns(2)

with col3:
    emp_length = st.selectbox("Employment Length", [
        '< 1 year', '1 year', '2 years', '3 years', '4 years',
        '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'
    ])
    home_ownership = st.selectbox("Home Ownership", ['MORTGAGE', 'RENT', 'OWN', 'OTHER', 'NONE', 'ANY'])
    verification_status = st.selectbox("Income Verification", ['Verified', 'Source Verified', 'Not Verified'])

with col4:
    open_acc = st.slider("Open Accounts", 1, 68, 10)
    pub_rec = st.slider("Public Records", 0, 21, 0)
    delinq_2yrs = st.slider("Delinquencies (Last 2 Years)", 0, 26, 0)

st.markdown("---")

# -------------------------
# Prediction
# -------------------------
if st.button("🦬 Generate Recommendation"):

    # Apply same preprocessing as notebook
    annual_inc_log = np.log1p(annual_inc)
    fico_avg = float(fico_score)
    dti_capped = min(dti, 38.48)
    revol_util_capped = min(revol_util, 100.0)

    input_df = pd.DataFrame({
        'loan_amnt': [loan_amnt],
        'int_rate': [int_rate],
        'grade': [grade],
        'emp_length': [emp_length],
        'home_ownership': [home_ownership],
        'verification_status': [verification_status],
        'purpose': [purpose],
        'dti': [dti_capped],
        'delinq_2yrs': [float(delinq_2yrs)],
        'open_acc': [float(open_acc)],
        'pub_rec': [float(pub_rec)],
        'revol_bal': [revol_bal],
        'revol_util': [revol_util_capped],
        'term_num': [term_num],
        'annual_inc_log': [annual_inc_log],
        'fico_avg': [fico_avg]
    })

    # One-hot encode
    cat_cols = ['grade', 'emp_length', 'home_ownership', 'verification_status', 'purpose']
    input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

    # Align columns with training data
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Scale
    input_scaled = pd.DataFrame(
        scaler.transform(input_df),
        columns=feature_columns
    )

    # Predict
    prob_fully_paid = classifier.predict_proba(input_scaled)[:, 0][0]
    prob_charged_off = classifier.predict_proba(input_scaled)[:, 1][0]
    pred_return = regressor.predict(input_scaled)[0]

    # Recommendation logic
    if prob_fully_paid >= 0.85:
        rec = "FUND"
        rec_class = "fund"
        rec_detail = "This application meets our lending criteria. Strong probability of full repayment."
    elif prob_fully_paid >= 0.70:
        rec = "REVIEW"
        rec_class = "review"
        rec_detail = "This application falls in a gray area. Recommend manual review before funding."
    else:
        rec = "DECLINE"
        rec_class = "decline"
        rec_detail = "This application presents elevated default risk. Not recommended for funding."

    # -------------------------
    # Display Results
    # -------------------------
    st.markdown("### 📊 Prediction Results")

    col5, col6, col7 = st.columns(3)
    with col5:
        st.markdown(f"""
            <div class='metric-card'>
                <p style='color:gray; margin:0; font-size:13px;'>P(Fully Paid)</p>
                <p style='color:#003366; font-size:28px; font-weight:bold; margin:0;'>{prob_fully_paid:.1%}</p>
            </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown(f"""
            <div class='metric-card'>
                <p style='color:gray; margin:0; font-size:13px;'>P(Charged Off)</p>
                <p style='color:#003366; font-size:28px; font-weight:bold; margin:0;'>{prob_charged_off:.1%}</p>
            </div>
        """, unsafe_allow_html=True)
    with col7:
        st.markdown(f"""
            <div class='metric-card'>
                <p style='color:gray; margin:0; font-size:13px;'>Predicted Return</p>
                <p style='color:#003366; font-size:28px; font-weight:bold; margin:0;'>{pred_return:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class='result-box {rec_class}'>
            <h2 style='color:white; margin:0;'>Recommendation: {rec}</h2>
            <p style='color:white; margin-top:8px;'>{rec_detail}</p>
        </div>
    """, unsafe_allow_html=True)

    # Probability bar chart
    st.markdown("### Probability Breakdown")
    chart_df = pd.DataFrame({
        'Outcome': ['Fully Paid', 'Charged Off'],
        'Probability': [prob_fully_paid, prob_charged_off]
    })

    probability_chart = alt.Chart(chart_df).mark_bar(
        cornerRadiusTopLeft=6,
        cornerRadiusTopRight=6,
        size=80
    ).encode(
        x=alt.X('Outcome:N', title=None, sort=['Fully Paid', 'Charged Off']),
        y=alt.Y('Probability:Q', title='Probability', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')),
        color=alt.Color(
            'Outcome:N',
            scale=alt.Scale(
                domain=['Fully Paid', 'Charged Off'],
                range=['#003865', '#E87722']
            ),
            legend=None
        ),
        tooltip=[
            alt.Tooltip('Outcome:N', title='Outcome'),
            alt.Tooltip('Probability:Q', title='Probability', format='.1%')
        ]
    ).properties(
        height=350
    )

    st.altair_chart(probability_chart, use_container_width=True)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("""
    <p class='footer'>
    🦬 Bucknell Lending Club | ANOP 330 Final Project | Lewisburg, PA
    </p>
""", unsafe_allow_html=True)
