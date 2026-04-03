import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Customer Churn Prediction", layout="wide", page_icon="👤")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(40, 48, 62) 0%, rgb(17, 24, 39) 90%);
    }
    
    h1 {
        background: linear-gradient(120deg, #f0abfc 0%, #fb7185 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeIn 1s ease-in;
    }
    h1::before { content: '📊 '; -webkit-text-fill-color: white; }
    
    p, label {
        color: #f1f5f9 !important;
    }
    
    /* Make input boxes sleek */
    input[type="number"], .stSelectbox > div > div {
        background: rgba(0,0,0,0.2) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        transition: border 0.3s ease;
    }
    
    input[type="number"]:focus, .stSelectbox > div > div:focus-within {
        border: 1px solid #fb7185 !important;
    }
    
    /* Large Red Predictive Button */
    .stButton > button {
        background: linear-gradient(135deg, #e11d48 0%, #db2777 100%);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.8rem;
        font-size: 18px;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(225, 29, 72, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 25px rgba(225, 29, 72, 0.6);
        color: white;
        border: none;
    }
    
    .stAlert {
        border-radius: 12px;
        animation: pulseFade 0.6s ease-out forwards;
    }
    
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    @keyframes pulseFade {
        0% { transform: scale(0.95); opacity: 0; }
        50% { transform: scale(1.02); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

st.title("Customer Attrition Analyst")
st.write("Advanced predictive analytics to evaluate the likelihood of immediate customer churn using Random Forest Logic.")

# Add a clean container layout
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        credit_score = st.number_input("Credit Score", 300, 850, 600)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 18, 100, 35)
        
    with col2:
        tenure = st.number_input("Tenure (Years)", 0, 10, 5)
        balance = st.number_input("Balance ($)", 0.0, 250000.0, 50000.0)
        num_products = st.number_input("Number of Products", 1, 4, 1)
        
    with col3:
        has_cr_card = st.selectbox("Has Credit Card?", [1, 0])
        is_active = st.selectbox("Is Active Member?", [1, 0])
        salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 60000.0)
        
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Generate Risk Assessment ⚡"):
    try:
        model = joblib.load('churn_prediction/churn_rf_model.pkl')
        scaler = joblib.load('churn_prediction/churn_scaler.pkl')
        expected_features = joblib.load('churn_prediction/churn_features.pkl')
        
        input_dict = {
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_products,
            'HasCrCard': has_cr_card,
            'IsActiveMember': is_active,
            'EstimatedSalary': salary
        }
        
        df = pd.DataFrame([input_dict])
        df = pd.get_dummies(df, columns=['Geography', 'Gender'])
        df = df.reindex(columns=expected_features, fill_value=0)
        
        scaled_data = scaler.transform(df)
        pred = model.predict(scaled_data)[0]
        
        if pred == 1:
            st.error("🚨 **CRITICAL RISK DETECTED:** Internal models forecast this client is highly likely to CHURN. Recommend immediate retention protocol.")
        else:
            st.success("✅ **STABLE PROFILE:** Client is modeled to stay. No immediate churn threat detected.")
        
    except FileNotFoundError:
         st.warning("Analytical Core Offline. Please compile models from shell terminal.")
