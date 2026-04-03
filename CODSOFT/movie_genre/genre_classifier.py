import pandas as pd
import string
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def clean_text(text):
    """Lowercases text and removes punctuation."""
    text = str(text).lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text

def train_genre_model(data_path="movies.csv"):
    print("Loading datasets...")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please place dataset in the folder.")
        return None, None
        
    df = pd.read_csv(data_path)
    
    # Adjust column names to lowercase to guarantee compatibility
    df.columns = [col.lower() for col in df.columns]
    
    # We predict 'genre' predominantly using 'description' or 'plot'
    if 'description' not in df.columns or 'genre' not in df.columns:
        print("Dataset must have 'description' and 'genre' columns.")
        return None, None
        
    df = df.dropna(subset=['description', 'genre'])
    
    print("Preprocessing movie descriptions...")
    df['desc_clean'] = df['description'].apply(clean_text)
    
    X = df['desc_clean']
    y = df['genre']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Vectorizing text descriptions with TF-IDF...")
    # Cap features to 5000 to keep it lightweight but effective
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("Training Logistic Regression Model (Multi-Class)...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    print("\n--- Evaluating Model ---")
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    
    # Save the artifacts
    joblib.dump(model, 'genre_model.pkl')
    joblib.dump(vectorizer, 'genre_vectorizer.pkl')
    print("\nModel and vectorizer saved as .pkl files.")
    return model, vectorizer

def predict_genre(plot_summary, model_path='genre_model.pkl', vectorizer_path='genre_vectorizer.pkl'):
    """Predicts genre for a new unseen movie plot."""
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    except FileNotFoundError:
        return "Model not trained yet. Run the script to generate .pkl"
        
    clean_plot = clean_text(plot_summary)
    plot_tfidf = vectorizer.transform([clean_plot])
    prediction = model.predict(plot_tfidf)[0]
    return prediction

if __name__ == "__main__":
    train_genre_model()
    
    # Test sample prediction
    sample_plot = "A group of intergalactic criminals must pull together to stop a fanatical warrior with plans to purge the universe."
    print("\nSample Prediction Test:")
    print(f"Plot: '{sample_plot}'")
    print(f"Predicted Genre: {predict_genre(sample_plot)}")
