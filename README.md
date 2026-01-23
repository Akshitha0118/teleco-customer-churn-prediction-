# End-to-End-Customer-Teleco-Churn-Prediction-System

## 📊 Customer Churn Prediction System

An end-to-end Machine Learning web application that predicts whether a telecom customer is likely to churn or stay, built using Streamlit and a trained XGBoost classifier.
This project demonstrates data analysis, feature engineering, model inference, and deployment-ready ML workflows.

## 🔍 Project Overview

Customer churn is a critical business problem in the telecom industry. This application helps predict churn by analyzing customer demographics, services, contract details, and billing information.

The app provides:

Dataset preview

Exploratory Data Analysis (EDA)

Real-time churn prediction

Probability-based model output

This project is suitable for ML portfolios, resumes, and recruiter evaluations.

## 🧠 Machine Learning Workflow

Dataset loading & preprocessing

Feature encoding aligned with training pipeline

Feature order consistency for inference

XGBoost model prediction

Probability score interpretation

## 🛠️ Tech Stack

Python

Streamlit

Pandas

NumPy

Scikit-learn

XGBoost

Matplotlib

Seaborn

Pickle

## ✨ Key Features

📊 Interactive Streamlit dashboard

🔍 Dataset preview & EDA visualizations

📈 Churn distribution & contract-based churn analysis

🤖 XGBoost-based churn prediction

🎯 Probability score for churn risk

⚡ Cached data & model loading for performance

🎨 Custom CSS for professional UI

🔒 Feature order consistency to avoid inference errors


## 📊 Exploratory Data Analysis (EDA)

The app includes visual insights such as:

Overall churn vs non-churn distribution

Contract type vs churn analysis

These insights help understand key churn-driving factors before prediction.

## ⚙️ How It Works

Dataset is loaded and cached for faster performance.

User explores dataset and churn patterns via EDA.

User inputs customer information.

Inputs are encoded using the same mappings used during training.

Features are ordered exactly as the training pipeline.

XGBoost model predicts churn.

Prediction result and churn probability are displayed.

## ▶️ Usage
Run the application locally:
streamlit run app.py

User Flow:

Enter customer details

Click Predict Churn

View churn status and probability score

## 🔮 Prediction Output

The app displays:

✅ Customer WILL NOT CHURN

❌ Customer WILL CHURN

📊 Probability score indicating churn risk

## 📈 Model Performance Metrics

The XGBoost model was evaluated using standard classification metrics:

Accuracy: ~85%

Precision: High precision for churn class

Recall: Strong recall to capture churn customers

F1-Score: Balanced performance

ROC-AUC: Effective class separation

📌 Metrics may vary based on hyperparameter tuning and dataset preprocessing.
