from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from keras.models import load_model
import pandas as pd
from baq.libs.utils.utilspredict import data_preprocessing, create_sequences, rolling_forecast
    # df = pd.read_csv('data/processed_data/data.csv')
    # X = create_sequences(df, target_column='pm2_5_(μg/m³)', sequence_length=24)
    # lstm_model = load_model('src/baq/libs/models/lstm_model_3.h5', compile=False)
    # y_pred = lstm_model.predict(X)
    # y_pred = y_pred.reshape(-1, 1)
    # print(y_pred)
app = FastAPI()

@app.post("/predict/onetime")
async def predict():
    # Load model
    try:
        lstm_model = load_model('src/baq/libs/models/lstm_model_3.h5', compile=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

    # Load Data
    try:
        #data = data_preprocessing('data/raw_data/baq_dataset.csv')
        data  = pd.read_csv('data/processed_data/data.csv')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading or preprocessing data: {e}")
    
    X = create_sequences(data, target_column='pm2_5_(μg/m³)', sequence_length=24)
    
    prediction = lstm_model.predict(X)
        
    
    return {"predicted_pm2_5": float(prediction[0][0])}

@app.post("/predict/rolling")
async def predict():
    # Load model
    try:
        lstm_model = load_model('src/baq/libs/models/lstm_model_3.h5', compile=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

    # Load Data
    try:
        #data = data_preprocessing('data/raw_data/baq_dataset.csv')
        data  = pd.read_csv('data/processed_data/data.csv')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading or preprocessing data: {e}")
    
    X = create_sequences(data, target_column='pm2_5_(μg/m³)', sequence_length=24)
    
    prediction = rolling_forecast(lstm_model, data, target_col='pm2_5_(μg/m³)', sequence_length=24, forecast_horizon=48)
        
    
    return (prediction.tolist())
