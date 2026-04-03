import pandas as pd
import string
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def clean_text(text):
    """Lowercases text and removes punctuation."""
    text = str(text).lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text

def train_spam_model(data_path="spam.csv"):
    """Trains the Spam SMS Detection model."""
    print("Loading data...")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please place dataset in the folder.")
        return None, None
        
    df = pd.read_csv(data_path, encoding='latin-1')
    
    # Handle common CODSOFT / Kaggle column names
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df.rename(columns={'v1': 'label', 'v2': 'text'})
        
    df = df[['label', 'text']].dropna()
    print("Preprocessing text...")
    df['text_clean'] = df['text'].apply(clean_text)
    
    # Encoding labels (spam=1, ham=0)
    X = df['text_clean']
    
    # Map words or handle if already numeric target types exist
    y = df['label'].map({'ham': 0, 'spam': 1})
    if y.isnull().all(): 
        y = df['label'] # Fallback
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("Training Naive Bayes Model...")
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    print("\n--- Evaluating Model ---")
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title("Spam Detection - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")
    print("Saved confusion matrix as 'confusion_matrix.png'")
    
    # Save the vectorizer and model for web app inference
    joblib.dump(model, 'spam_model.pkl')
    joblib.dump(vectorizer, 'spam_vectorizer.pkl')
    print("Model and vectorizer saved successfully.")
    
    return model, vectorizer

def predict_message(message, model_path='spam_model.pkl', vectorizer_path='spam_vectorizer.pkl'):
    """Predicts whether a single given message is Spam or Ham."""
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    except FileNotFoundError:
        return "Model not found. Please train the model first."
        
    msg_clean = clean_text(message)
    msg_tfidf = vectorizer.transform([msg_clean])
    prediction = model.predict(msg_tfidf)[0]
    return "Spam" if prediction == 1 else "Ham"

if __name__ == "__main__":
    train_spam_model()
    # Test the predict function
    sample_msg = "WINNER!! As a valued network customer you have been selected to receive a $900 prize reward!"
    print(f"\nSample Prediction Test:")
    print(f"Message: '{sample_msg}'")
    print(f"Result: {predict_message(sample_msg)}")
