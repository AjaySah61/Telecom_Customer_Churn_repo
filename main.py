from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pickle
import pandas as pd
from pydantic import BaseModel, Field
from typing import List

app = FastAPI()

model = pickle.load(open("telecom_customer_churn_logistic_model.pkl", "rb"))
'''SeniorCitizen', 'Partner', 'Dependents', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure', 'MonthlyCharges', 'TotalCharges'''
class UserInput(BaseModel):
    SeniorCitizen: str = Field(default="Yes", description="Yes/No")
    Partner: str = Field(default="Yes", description="Yes/No")
    Dependents: str = Field(default="Yes", description="Yes/No")
    MultipleLines: str = Field(default="Yes", description="Yes/No/No phone service")
    InternetService: str = Field(default="DSL", description="DSL/Fiber optic/No")
    OnlineSecurity: str = Field(default="Yes", description="Yes/No/No internet service")
    OnlineBackup: str = Field(default="Yes", description="Yes/No/No internet service")
    DeviceProtection: str = Field(default="Yes", description="Yes/No/No internet service")
    TechSupport: str = Field(default="Yes", description="Yes/No/No internet service")
    StreamingTV: str = Field(default="Yes", description="Yes/No/No internet service")
    StreamingMovies: str = Field(default="Yes", description="Yes/No/No internet service")
    Contract: str = Field(default="Month-to-month", description="Month-to-month/One year/Two year")
    PaperlessBilling: str = Field(default="Yes", description="Yes/No")
    PaymentMethod: str = Field(default="Bank transfer (automatic)", description="Bank transfer (automatic)/Credit card (automatic)/Electronic check/Mailed check")
    tenure: int = Field(default=1, description="Enter Int(0, 72)")
    MonthlyCharges: float = Field(default=39.65, description="Enter Float(18.00, 118.75)")
    TotalCharges: float = Field(default=18.80, description="Enter Float(18.80, 8684.80)")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Customer Churn Prediction</title>
    </head>
    <body>
        <h1>Welcom to Machine Learning Model</h1>
        <p>To run this model, please click on given link below!</p>
        <a href="/docs" style="font-size: 20px; color: blue; font-weight: bold;">Open API Documentation (Docs)</a>
        
    </body>
    </html>
    """

@app.post("/predict")
def predict(data: List[UserInput]):
    # 1. Convert input to DataFrame
    # input_data = data.model_dump()
    input_df = pd.DataFrame([item.model_dump() for item in data])
        
    try:
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df) * 100
        # result = int(predictions[0])

        results = []
        for i in range(len(predictions)):
            pred = int(predictions[i])
            conf = float(probabilities[i][pred])
            results.append({
                "prediction": "Churn True" if pred == 1 else "Churn False",
                "confidence": round(conf, 2)
            })
        
        return {
            "results": results
        }
    except Exception as e:
        return {"error": str(e), "message": "For any issue report on this email- ajaysah.jobs.20@gmail.com"}