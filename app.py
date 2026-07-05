import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Scorecard",
    page_icon="🏦",
    layout="centered"
)

# ── Load model and artifacts ──────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_columns.pkl', 'rb') as f:
        features = pickle.load(f)
    with open('explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    return model, features, explainer

model, feature_columns, explainer = load_artifacts()

# ── Header ────────────────────────────────────────────────────
st.title("🏦 Credit Risk Scorecard")
st.markdown("Enter loan applicant details to predict default probability and explain the decision.")
st.divider()

# ── Input form ────────────────────────────────────────────────
st.subheader("Applicant Details")

col1, col2 = st.columns(2)

with col1:
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=40000, value=10000, step=500)
    int_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=30.0, value=12.0, step=0.1)
    annual_inc = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=60000, step=1000)
    dti = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
    fico_range_low = st.number_input("FICO Score", min_value=580, max_value=850, value=700, step=5)

with col2:
    installment = st.number_input("Monthly Installment ($)", min_value=10, max_value=1500, value=300, step=10)
    open_acc = st.number_input("Open Credit Accounts", min_value=0, max_value=50, value=8, step=1)
    pub_rec = st.number_input("Public Records", min_value=0, max_value=10, value=0, step=1)
    revol_bal = st.number_input("Revolving Balance ($)", min_value=0, max_value=100000, value=5000, step=500)
    revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)

col3, col4 = st.columns(2)

with col3:
    total_acc = st.number_input("Total Credit Accounts", min_value=0, max_value=100, value=15, step=1)
    emp_length = st.selectbox("Employment Length", 
        options=[0,1,2,3,4,5,6,7,8,9,10,11],
        format_func=lambda x: {
            0: "Unknown", 1: "< 1 year", 2: "1 year", 3: "2 years",
            4: "3 years", 5: "4 years", 6: "5 years", 7: "6 years",
            8: "7 years", 9: "8 years", 10: "9 years", 11: "10+ years"
        }[x],
        index=6
    )
    term = st.selectbox("Loan Term", options=[36, 60], index=0)

with col4:
    home_ownership = st.selectbox("Home Ownership", 
        options=["MORTGAGE", "RENT", "OWN", "OTHER", "NONE", "ANY"],
        index=0
    )
    purpose = st.selectbox("Loan Purpose",
        options=["debt_consolidation", "credit_card", "home_improvement", 
                 "major_purchase", "medical", "small_business", "car",
                 "vacation", "moving", "house", "renewable_energy", 
                 "educational", "wedding", "other"],
        index=0
    )

st.divider()

# ── Predict ───────────────────────────────────────────────────
if st.button("Score Applicant", type="primary", use_container_width=True):

    # Build input row with all zeros
    input_dict = {col: 0 for col in feature_columns}

    # Fill numeric features
    input_dict['loan_amnt'] = loan_amnt
    input_dict['int_rate'] = int_rate
    input_dict['installment'] = installment
    input_dict['annual_inc'] = annual_inc
    input_dict['dti'] = dti
    input_dict['fico_range_low'] = fico_range_low
    input_dict['open_acc'] = open_acc
    input_dict['pub_rec'] = pub_rec
    input_dict['revol_bal'] = revol_bal
    input_dict['revol_util'] = revol_util
    input_dict['total_acc'] = total_acc
    input_dict['emp_length'] = emp_length
    input_dict['term'] = term

    # One-hot encode home_ownership
    ownership_col = f'home_ownership_{home_ownership}'
    if ownership_col in input_dict:
        input_dict[ownership_col] = True

    # One-hot encode purpose
    purpose_col = f'purpose_{purpose}'
    if purpose_col in input_dict:
        input_dict[purpose_col] = True

    # Create dataframe
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_columns]

    # Predict
    prob = model.predict_proba(input_df)[0][1]
    threshold = 0.57
    decision = "REJECT" if prob >= threshold else "APPROVE"

    # ── Results ───────────────────────────────────────────────
    st.subheader("Decision")

    if decision == "APPROVE":
        st.success(f"✅ APPROVED — Default Probability: {prob:.1%}")
    else:
        st.error(f"❌ REJECTED — Default Probability: {prob:.1%}")

    # Risk meter
    st.progress(float(prob), text=f"Risk Score: {prob:.1%} (Threshold: {threshold:.0%})")

    st.divider()

    # ── SHAP explanation ──────────────────────────────────────
    st.subheader("Why this decision?")

    shap_values = explainer.shap_values(input_df)
    shap_explanation = explainer(input_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_explanation[0], show=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption("Blue bars reduce default risk. Red bars increase default risk. The final score determines approval.")

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption("Built by Dhriti Jengiti · Georgia Tech CS + Finance · Credit Risk Scorecard Project")
