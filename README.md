# Customer Churn Prediction API

## Objective
The objective of this project is to build an end-to-end machine learning solution that predicts whether a customer is likely to churn, starting from raw CSV data and ending with a deployable REST API.  
The project demonstrates data preprocessing, model selection, evaluation, and deployment using real-world structured data.

---

## Introduction
Customer churn prediction is a common and critical business problem in subscription-based industries. Retaining existing customers is often more cost-effective than acquiring new ones.

This project implements a machine learning pipeline that:
- Cleans and preprocesses customer data
- Trains and evaluates multiple classification models
- Selects the best-performing model using appropriate metrics
- Deploys the trained model as a REST API using FastAPI

The solution is designed to be simple, explainable, and production-ready.

---

## Features
- Handles missing values in numerical and categorical data
- Encodes categorical variables safely using One-Hot Encoding
- Scales numerical features where required
- Trains and compares multiple ML models
- Uses F1-score for model selection due to class imbalance
- Deploys predictions through a FastAPI endpoint
- Accepts and returns data in JSON format
- Includes a trained model artifact for direct execution

---

## Tech Stack
- **Programming Language:** Python 3.12
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Model Serialization:** Joblib
- **API Framework:** FastAPI
- **ASGI Server:** Uvicorn
- **Development Environment:** VS Code, Virtual Environment (venv)

---



## Project Structure
mini_ai_solution/
│
├── data/
│ └── churn.csv
│
├── notebooks/
│ └── 01_eda_customer_churn.ipynb
│
├── src/
│ ├── config.py
│ ├── preprocessing.py
│ └── train.py
│
├── app.py
├── model.joblib
├── requirements.txt
├── README.md
├── .gitignore
└── myenv/ (local virtual environment, not pushed)




## Setup Instructions

### 1. Create Virtual Environment
python -m venv myenv


2. Activate Virtual Environment

Windows
myenv\Scripts\activate


Linux / macOS
source myenv/bin/activate
3. Install Dependencies



pip install -r requirements.txt
Model Training
Train the machine learning models and generate the model artifact:


python src/train.py


This script:

Applies preprocessing using pipelines

Trains Logistic Regression and Random Forest models

Evaluates them using F1-score

Saves the best-performing model as model.joblib

API Deployment
Start the FastAPI server:


uvicorn app:app --reload


Access Swagger UI at:
http://127.0.0.1:8000/docs



API Usage
Endpoint
bash

POST /predict
Sample Request (JSON)
json

{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 5,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 80,
  "TotalCharges": 400
}
Sample Response
json
     
{
  "churn_prediction": 1,
  "churn_probability": 0.76
}
Model Explanation (Summary)
Logistic Regression and Random Forest models were evaluated.

F1-score was used as the primary metric due to class imbalance.

Logistic Regression performed better without hyperparameter tuning.

The model is interpretable and suitable for business decision-making.

A detailed explanation of model logic and feature impact is documented separately.



Conclusion
This project successfully demonstrates an end-to-end machine learning workflow, from data preprocessing and model training to deployment and API testing. The solution is clean, explainable, and aligned with real-world industry practices. It can be extended further with advanced optimization, monitoring, and explainability techniques.

Author
Renjith R