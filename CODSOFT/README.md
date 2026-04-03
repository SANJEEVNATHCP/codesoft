# CODSOFT ML Internship Projects

This repository contains the completion of 3 core machine learning tasks commonly assigned during the CODSOFT ML Internship:
1. **Spam SMS Detection** (NLP)
2. **Movie Genre Classification** (NLP)
3. **Customer Churn Prediction** (Classification)

It additionally includes a unified **Streamlit Web App** to interactively showcase and test all three trained models in a real-world dashboard setting.

## 📁 Repository Structure
```text
CODSOFT/
│
├── spam_detection/
│   ├── spam_classifier.py      # Preprocessing, TF-IDF, Naive Bayes
│   └── spam.csv                # (Run generate_dummy_data.py to mock if missing)
│
├── movie_genre/
│   ├── genre_classifier.py     # NLP preprocessing, TF-IDF, Logistic Regression
│   └── movies.csv              # (Run generate_dummy_data.py to mock if missing)
│
├── churn_prediction/
│   ├── churn_classifier.py     # Missing value handling, One-hot, Random Forest
│   └── Churn_Modelling.csv     # (Run generate_dummy_data.py to mock if missing)
│
├── app.py                      # Interactive Streamlit Web Interface
├── generate_dummy_data.py      # Auto-generates placeholder CSV datasets (Perfect for out-of-the-box testing)
├── requirements.txt            # Python environment dependencies
└── README.md                   # Documentation File
```

## ✨ Features
- **Proper Data Preprocessing:** Robust handling of text sanitization (NLP), missing values, one-hot encoding, and numerical scaling.
- **Accurate Model Training:** Applies Scikit-Learn standard models like `MultinomialNB`, `LogisticRegression`, and `RandomForestClassifier`.
- **In-depth Evaluation:** Evaluates systems through custom code measuring `Accuracy`, printing `Classification Reports`, and generating Matplotlib `Confusion Matrices` & `Feature Importance` plots.
- **Interactive UI (Streamlit):** Easily showcase your skills through an interactive web-based portfolio.

## 🛠️ Installation & Execution Steps

**1. Install Dependencies:**
Ensure Python 3.8+ is installed. Run the following command in the terminal inside this directory to install library requirements:
```bash
pip install -r requirements.txt
```

**2. Setup Data (Optional shortcut step):**
If you already possess the precise CODSOFT CSV files, place them in their respective prediction folders (e.g. `movies.csv`, `spam.csv`, `Churn_Modelling.csv`).
Alternatively, if you want this to **run out-of-the-box immediately**, generate standard structural datasets by running:
```bash
python generate_dummy_data.py
```

**3. Train the Models:**
Navigate into each directory and execute the underlying scripts directly to preprocess data, output analytics, and save the binary model components (`.pkl` files) utilized by the Streamlit App.
```bash
cd spam_detection
python spam_classifier.py
cd ..

cd movie_genre
python genre_classifier.py
cd ..

cd churn_prediction
python churn_classifier.py
cd ..
```

**4. Run the Streamlit Application (Demo Mode):**
To showcase your application and test text entries, launch the app:
```bash
streamlit run app.py
```

## 📸 Demo Explanation
Once Streamlit runs, use the Left Sidebar to switch swiftly between tasks:
- **Task 1:** Let's you paste an SMS string and informs you graphically if it registers as 'Spam'.
- **Task 2:** Write a quick plot outline and predict a corresponding genre instantly using TF-IDF logic.
- **Task 3:** Adjust mock-customer information inputs (Balance, Country, Salary, Age) securely within standard boundary layers to view instantaneous retention vs. churn forecasting.

Made to be copied, implemented, styled, and submitted accurately to reflect senior expertise within the machine learning spectrum.
