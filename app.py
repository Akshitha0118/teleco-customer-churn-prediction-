import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# =====================================
# CUSTOM CSS
# =====================================
st.markdown("""
<style>
body { background-color: #f4f6f9; }
.main { background-color: white; padding: 25px; border-radius: 12px; }
h1, h2, h3 { color: #2c3e50; }
.stButton button {
    background-color: #1abc9c;
    color: white;
    font-size: 18px;
    border-radius: 8px;
    padding: 8px 25px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main'>", unsafe_allow_html=True)

# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_data():
    return pd.read_csv("Telco-Customer-Churn.csv")

df = load_data()

# =====================================
# LOAD MODEL
# =====================================
@st.cache_resource
def load_model():
    with open("final_xgb_classifier.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# =====================================
# MODEL FEATURE ORDER (CRITICAL)
# =====================================
FEATURE_ORDER = [
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod',
    'MonthlyCharges',
    'TotalCharges',
    'tenure_group'
]

# =====================================
# TITLE
# =====================================
st.title("📊 Customer Churn Prediction System")
st.write("Predict whether a customer will **Churn or Not** using XGBoost.")

# =====================================
# DATASET PREVIEW
# =====================================
st.header("📁 Dataset Preview")
st.dataframe(df.head())

# =====================================
# EDA
# =====================================
st.header("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.countplot(x="Churn", data=df, ax=ax)
    st.pyplot(fig)

with col2:
    fig2, ax2 = plt.subplots()
    sns.countplot(x="Contract", hue="Churn", data=df, ax=ax2)
    st.pyplot(fig2)

# =====================================
# USER INPUT (IMPORTANT FEATURES ONLY)
# =====================================
st.header("🧾 Enter Customer Details")

Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
PaymentMethod = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0)
tenure_group = st.selectbox("Tenure Group", [0, 1, 2, 3, 4])

# =====================================
# ENCODING (MATCH TRAINING)
# =====================================
encoding_maps = {
    "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2},
    "OnlineSecurity": {"No": 0, "Yes": 1, "No internet service": 2},
    "TechSupport": {"No": 0, "Yes": 1, "No internet service": 2},
    "PaymentMethod": {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3
    },
    "PaperlessBilling": {"No": 0, "Yes": 1}
}

# =====================================
# INPUT DATA (ALL FEATURES)
# =====================================
input_data = {
    'gender': 0,
    'SeniorCitizen': 0,
    'Partner': 1,
    'Dependents': 1,
    'PhoneService': 1,
    'MultipleLines': 0,
    'InternetService': encoding_maps["InternetService"][InternetService],
    'OnlineSecurity': encoding_maps["OnlineSecurity"][OnlineSecurity],
    'OnlineBackup': 1,
    'DeviceProtection': 1,
    'TechSupport': encoding_maps["TechSupport"][TechSupport],
    'StreamingTV': 0,
    'StreamingMovies': 0,
    'Contract': encoding_maps["Contract"][Contract],
    'PaperlessBilling': encoding_maps["PaperlessBilling"][PaperlessBilling],
    'PaymentMethod': encoding_maps["PaymentMethod"][PaymentMethod],
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges,
    'tenure_group': tenure_group
}

input_df = pd.DataFrame([input_data])
input_df = input_df[FEATURE_ORDER]  # 🔥 exact order

# =====================================
# PREDICTION
# =====================================
st.header("🔮 Prediction Result")

if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"❌ Customer WILL CHURN\n\nProbability: {probability:.2%}")
    else:
        st.success(f"✅ Customer WILL NOT CHURN\n\nProbability: {probability:.2%}")

st.markdown("</div>", unsafe_allow_html=True)

