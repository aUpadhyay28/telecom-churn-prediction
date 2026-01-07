🔍Overview
   Customer churn is a major challenge in the telecom industry. This project builds an end-to-end Machine Learning–based system to predict whether a customer is likely to leave a telecom service, enabling proactive retention strategies.
   The system includes data preprocessing, feature engineering, multiple ML models, model evaluation, and an interactive Streamlit dashboard for real-time predictions.

🎯 Problem Statement
    Telecom companies lose significant revenue due to customer churn.
    The objective of this project is to predict customer churn in advance using historical customer data such as demographics, service usage, and billing details.

🚀 Key Features
   1. Single Customer Churn Prediction
   2. Batch Prediction using CSV Upload
   3. Churn Analytics & Visualization Dashboard
   4. Multiple ML Models Comparison
   5. Model Confidence & Risk Level Estimation
   6. Feature Engineering for Better Accuracy

🧠Machine Learning Models Used
   
   1.Logistic Regression
   2.Decision Tree Classifier
   3.Support Vector Machine (SVM)
   4.Evaluation Metrics:
     a.Accuracy
     b.Precision
     c.Recall
     d.F1-Score

📁 Project Structure
    
    telecom-churn-prediction/
    │
    ├── app.py                # Streamlit application
    ├── data_processor.py     # Data cleaning & preprocessing
    ├── model_trainer.py      # Model training & evaluation
    ├── predictor.py          # Prediction logic
    ├── visualizations.py     # Charts & dashboards
    ├── data/                 # Dataset
    ├── .gitignore
    └── README.md
