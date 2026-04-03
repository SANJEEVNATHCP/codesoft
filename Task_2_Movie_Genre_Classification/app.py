import streamlit as st
import joblib
import string

st.set_page_config(page_title="Movie Genre Classification", layout="centered", page_icon="🎬")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #020617 0%, #172554 100%);
    }
    
    h1 {
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeInDown 0.8s ease-out;
    }
    h1::before { content: '🎬 '; -webkit-text-fill-color: white; }
    
    p, li, label {
        color: #e0f2fe !important;
    }
    
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        color: white;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:hover, .stTextArea textarea:focus {
        border-color: #38bdf8;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #0ea5e9 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 0;
        font-weight: 600;
        letter-spacing: 1px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(14, 165, 233, 0.5);
        color: white;
        border: none;
    }
    
    .stAlert {
        border-left: 5px solid #38bdf8;
        animation: slideInRight 0.5s ease-out forwards;
    }
    
    @keyframes fadeInDown {
        0% { opacity: 0; transform: translateY(-20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideInRight {
        0% { transform: translateX(30px); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

st.title("Movie Genre Predictor")
st.write("Artificial intelligence analysis to instantly determine film genres based on semantic plot interpretation.")

plot = st.text_area("Cinematic Plot Analysis:", "A lone survivor forms an unlikely alliance with a rogue AI to traverse a devastated earth.", height=150)

if st.button("Predict Genre 🔮"):
    try:
        model = joblib.load('movie_genre/genre_model.pkl')
        vectorizer = joblib.load('movie_genre/genre_vectorizer.pkl')
        
        clean_plot = plot.lower()
        clean_plot = "".join([char for char in clean_plot if char not in string.punctuation])
        
        vec_plot = vectorizer.transform([clean_plot])
        pred = model.predict(vec_plot)[0]
        st.success(f"### ✨ Determined Genre: **{pred.upper()}**")
    except FileNotFoundError:
        st.warning("Prediction Pipeline Offline. Ensure Model `.pkl` generation through terminal.")
