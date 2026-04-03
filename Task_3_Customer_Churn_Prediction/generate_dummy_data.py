import pandas as pd
import numpy as np
import os

os.makedirs('spam_detection', exist_ok=True)
os.makedirs('movie_genre', exist_ok=True)
os.makedirs('churn_prediction', exist_ok=True)

# 1. Spam dataset dummy
print("Generating dummy spam dataset...")
spam_data = pd.DataFrame({
    'v1': ['ham', 'spam', 'ham', 'spam'] * 250,
    'v2': ['Hi how are you', 'WINNER claim your prize now', 'Call me later', 'URGENT your account is locked'] * 250
})
spam_data.to_csv('spam_detection/spam.csv', index=False)

# 2. Movie Genre dataset dummy
print("Generating dummy movie genre dataset...")
movie_data = pd.DataFrame({
    'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D'] * 250,
    'genre': ['Action', 'Comedy', 'Horror', 'Romance'] * 250,
    'description': [
        'A brave hero fights space aliens to save the earth', 
        'Funny guy does hilarious things in an awkward high school', 
        'Scary monster attacks a quiet small town at midnight', 
        'Two people fall in deep love against all odds'
    ] * 250
})
movie_data.to_csv('movie_genre/movies.csv', index=False)

# 3. Churn dataset dummy
print("Generating dummy churn dataset...")
churn_data = pd.DataFrame({
    'RowNumber': range(1, 1001),
    'CustomerId': np.random.randint(15000000, 15999999, 1000),
    'Surname': ['Smith', 'Johnson', 'Brown', 'Lee', 'Garcia'] * 200,
    'CreditScore': np.random.randint(600, 800, 1000),
    'Geography': np.random.choice(['France', 'Spain', 'Germany'], 1000),
    'Gender': np.random.choice(['Male', 'Female'], 1000),
    'Age': np.random.randint(20, 60, 1000),
    'Tenure': np.random.randint(0, 10, 1000),
    'Balance': np.random.uniform(1000, 150000, 1000),
    'NumOfProducts': np.random.randint(1, 4, 1000),
    'HasCrCard': np.random.randint(0, 2, 1000),
    'IsActiveMember': np.random.randint(0, 2, 1000),
    'EstimatedSalary': np.random.uniform(30000, 100000, 1000),
    'Exited': np.random.randint(0, 2, 1000)
})
churn_data.to_csv('churn_prediction/Churn_Modelling.csv', index=False)

print("Dummy data generated successfully!")
