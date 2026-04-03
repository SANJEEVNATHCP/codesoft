import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(page_title="CODSOFT ML Projects", layout="wide", page_icon="🚀")

st.sidebar.title("CODSOFT ML Projects")
project = st.sidebar.radio("Select a Project:", 
                           ["Home", 
                            "1. Spam SMS Detection", 
                            "2. Movie Genre Classification", 
                            "3. Customer Churn Prediction"])

if project == "Home":
    st.title("Welcome to CODSOFT Machine Learning Projects")
    st.write("This application demonstrates three machine learning projects:")
    st.markdown("""
    - **Spam SMS Detection:** Classifies messages as Spam or Ham using Naive Bayes and TF-IDF.
    - **Movie Genre Classification:** Combines Natural Language Processing and Logistic Regression to predict movie genres based on plots.
    - **Customer Churn Prediction:** Predicts whether a customer will leave the bank using a Random Forest Classifier.
    """)
    st.info("👈 Please select a project from the sidebar to test the predictions.")

elif project == "1. Spam SMS Detection":
    st.title("📩 Spam SMS Detection")
    st.write("Enter a text message to check if it represents 'Spam' or 'Ham'.")
    
    message = st.text_area("Message:", "Congratulations! You've won a $1000 Walmart gift card. Click here to claim.")
    
    if st.button("Predict Spam/Ham"):
        try:
            model = joblib.load('spam_detection/spam_model.pkl')
            vectorizer = joblib.load('spam_detection/spam_vectorizer.pkl')
            
            import string
            clean_msg = message.lower()
            clean_msg = "".join([char for char in clean_msg if char not in string.punctuation])
            
            vec_msg = vectorizer.transform([clean_msg])
            pred = model.predict(vec_msg)[0]
            
            if pred == 1:
                st.error("🚨 This message is predicted as: SPAM")
            else:
                st.success("✅ This message is predicted as: HAM (Safe)")
        except FileNotFoundError:
            st.warning("Model not found. Please navigate to `spam_detection` and run `spam_classifier.py` first to train the model.")

elif project == "2. Movie Genre Classification":
    st.title("🎬 Movie Genre Classification")
    st.write("Enter a movie plot description to predict its genre.")
    
    plot = st.text_area("Movie Plot:", "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.")
    
    if st.button("Predict Genre"):
        try:
            model = joblib.load('movie_genre/genre_model.pkl')
            vectorizer = joblib.load('movie_genre/genre_vectorizer.pkl')
            
            import string
            clean_plot = plot.lower()
            clean_plot = "".join([char for char in clean_plot if char not in string.punctuation])
            
            vec_plot = vectorizer.transform([clean_plot])
            pred = model.predict(vec_plot)[0]
            st.success(f"🎥 Predicted Genre: **{pred.title()}**")
        except FileNotFoundError:
            st.warning("Model not found. Please navigate to `movie_genre` and run `genre_classifier.py` first to train the model.")

elif project == "3. Customer Churn Prediction":
    st.title("👤 Customer Churn Prediction")
    st.write("Provide customer details to predict if they will churn (leave) or stay.")
    
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
        
    if st.button("Predict Churn"):
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
                st.error("⚠️ Prediction: This customer is highly likely to CHURN (leave).")
            else:
                st.success("✅ Prediction: This customer is expected to STAY (No Churn).")
            
        except FileNotFoundError:
             st.warning("Model not found. Please navigate to `churn_prediction` and run `churn_classifier.py` first to train the model.")
