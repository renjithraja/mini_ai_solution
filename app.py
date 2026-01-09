from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a customer is likely to churn",
    version="1.0"
)

model = joblib.load("model.joblib")


@app.post("/predict")
def predict_churn(customer_data: dict):
    """
    Accepts customer details as JSON and returns churn prediction.
    """
    df = pd.DataFrame([customer_data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(probability, 2)
    }
