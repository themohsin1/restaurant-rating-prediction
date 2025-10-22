
# src/api.py
from fastapi import FastAPI
import joblib
import pandas as pd

# Initialize app
app = FastAPI(title="Restaurant Rating Predictor", version="1.0")

# Load the trained pipeline (FeatureBuilder + Model)
artifacts = joblib.load("models/restaurant_pipeline.pkl")
builder = artifacts["builder"]
model = artifacts["model"]

@app.get("/")
def home():
    """Simple welcome route."""
    return {"message": "Welcome to the Restaurant Rating Prediction API 🚀"}

@app.post("/predict")
def predict(data: dict):
    """
    Takes restaurant details as JSON and returns a predicted rating.
    Expected JSON keys:
    online_order, book_table, location, rest_type, cuisines, approx_cost
    """
    # Convert JSON to DataFrame
    df = pd.DataFrame([{
        "online_order": data["online_order"],
        "book_table": data["book_table"],
        "location": data["location"],
        "rest_type": data["rest_type"],
        "cuisines": data["cuisines"],
        "approx_cost(for two people)": data["approx_cost"]
    }])

    # Transform features
    X = builder.transform(df)

    # Predict rating
    pred = model.predict(X).clip(1.0, 5.0)
    return {"predicted_rating": round(float(pred[0]), 2)}
