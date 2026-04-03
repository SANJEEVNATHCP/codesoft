import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def preprocess_churn_data(df):
    """
    Handles missing values, encoding, and preparing variables.
    Tailored to handle typical Customer Churn constraints securely.
    """
    # Drop irrelevant identifier columns if they exist
    cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Deal with missing values: Median for numericals, mode for objects
    df = df.fillna(df.median(numeric_only=True))
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    # Isolate targets securely (Assume standard name 'Exited' or 'Churn')
    target_col = 'Exited' if 'Exited' in df.columns else 'Churn'
    if target_col not in df.columns:
        return None, None, None
        
    y = df[target_col]
    # Ensure Binary Format (Example: Yes/No -> 1/0)
    if y.dtype == 'object':
        y = y.map({'Yes': 1, 'No': 0, '1': 1, '0': 0})
        
    X = df.drop(columns=[target_col])
    
    # Feature Encoding (One-Hot for variables like Geography, Male/Female)
    categorical_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    return X, y, X.columns

def train_churn_models(data_path="Churn_Modelling.csv"):
    print("Loading data...")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    X, y, feature_names = preprocess_churn_data(df)
    
    if X is None:
        print("Target categorical column ('Exited' or 'Churn') not found in dataset!")
        return
        
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Applying Feature Scaling for uniform modeling metrics...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n--- Training Model 1: Logistic Regression ---")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_pred)
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    
    print("\n--- Training Model 2: Random Forest (Primary) ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_pred))
    
    # Generating Feature Importance visually to analyze retention factors
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Customer Churn - Random Forest Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("Saved feature importances plot to 'feature_importance.png'")
    
    # Save the superior model (Random Forest) to act as standard implementation
    joblib.dump(rf_model, 'churn_rf_model.pkl')
    joblib.dump(scaler, 'churn_scaler.pkl')
    joblib.dump(list(feature_names), 'churn_features.pkl') # Vital for reconstructing shape constraints
    print("Model, scaler, and expected feature map successfully saved.")
    
if __name__ == "__main__":
    train_churn_models()
