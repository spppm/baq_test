from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from baq.libs.utils.utilsAPI import load_model
from baq.libs.utils.utilspredict import data_preprocessing

app = FastAPI()

@app.post("/predict")
async def predict():
    # Load model
    try:
        lstm_model = load_model('src/baq/libs/models/lstm_model_3.h5', compile=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

    # Preprocess data
    try:
        data = data_preprocessing('data/raw_data/baq_dataset.csv')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading or preprocessing data: {e}")
    
    # Predict
    try:
        prediction = lstm_model.predict(data)
        if prediction.size == 0:
            raise HTTPException(status_code=500, detail="Empty prediction result.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    
    return {"predicted_pm2_5": float(prediction[0][0])}
