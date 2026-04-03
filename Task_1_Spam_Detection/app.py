import streamlit as st
import joblib
import string

st.set_page_config(page_title="Spam SMS Detection", layout="centered", page_icon="📩")

# UI Enhancements
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    }
    
    h1 {
        background: -webkit-linear-gradient(45deg, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeInDown 0.8s ease-out;
    }
    h1::before { content: '📩 '; -webkit-text-fill-color: white; }
    
    p, li, label {
        color: #e2e8f0 !important;
    }
    
    /* Clean glassmorphism text-area */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: white;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:hover, .stTextArea textarea:focus {
        border-color: #818cf8;
        box-shadow: 0 0 15px rgba(129, 140, 248, 0.3);
    }
    
    /* Premium Button Hover & Gradient */
    .stButton > button {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 0;
        font-weight: 600;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(124, 58, 237, 0.4);
        color: white;
        border: none;
    }
    
    .stAlert {
        animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
        border-radius: 12px;
        border: none;
    }
    
    /* Keyframe Animations */
    @keyframes fadeInDown {
        0% { opacity: 0; transform: translateY(-20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes popIn {
        0% { transform: scale(0.9); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

st.title("Spam SMS Detection")
st.write("Enter a text message to instantly check if it represents **'Spam'** or **'Ham'**.")

message = st.text_area("Secure Message Scanner:", "Congratulations! You've won a $1000 Walmart gift card. Click here to claim.", height=150)

if st.button("Authenticate & Predict"):
    try:
        model = joblib.load('spam_detection/spam_model.pkl')
        vectorizer = joblib.load('spam_detection/spam_vectorizer.pkl')
        
        clean_msg = message.lower()
        clean_msg = "".join([char for char in clean_msg if char not in string.punctuation])
        
        vec_msg = vectorizer.transform([clean_msg])
        pred = model.predict(vec_msg)[0]
        
        if pred == 1:
            st.error("🚨 **ALERT!** This message is classified as highly dangerous **SPAM**.")
        else:
            st.success("✅ **SAFE!** This message is verified as authentic **HAM**.")
    except FileNotFoundError:
        st.warning("Model not found. Please train the model from your terminal first.")
