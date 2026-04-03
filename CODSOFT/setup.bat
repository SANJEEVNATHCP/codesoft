@echo off
echo Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error during pip install.
    exit /b %errorlevel%
)

echo Generating dummy data...
python generate_dummy_data.py
if %errorlevel% neq 0 (
    echo Error generating data.
    exit /b %errorlevel%
)

echo Training Spam detection model...
cd spam_detection
python spam_classifier.py
cd ..

echo Training Movie Genre model...
cd movie_genre
python genre_classifier.py
cd ..

echo Training Churn Prediction model...
cd churn_prediction
python churn_classifier.py
cd ..

echo All models trained! Launching the Streamlit Web Application...
streamlit run app.py
