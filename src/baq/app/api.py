from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from keras.models import load_model
import pandas as pd
from baq.libs.utils.utilspredict import data_preprocessing, create_sequences, rolling_forecast

app = FastAPI()

@app.post("/predict/onetime")
async def predict_onetime():
    try:
        lstm_model = load_model('src/baq/libs/models/lstm_model_3.h5', compile=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

    try:
        data  = pd.read_csv('data/processed_data/data.csv')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading or preprocessing data: {e}")
    
    X = create_sequences(data, target_column='pm2_5_(μg/m³)', sequence_length=24)
    prediction = lstm_model.predict(X).reshape(-1, 1)

    return {"predictions": prediction.flatten().tolist()}

@app.post("/predict/rolling")
async def predict_rolling():
    try:
        lstm_model = load_model('src/baq/libs/models/lstm_model_3.h5', compile=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

    try:
        data  = pd.read_csv('data/processed_data/data.csv')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading or preprocessing data: {e}")
    
    prediction = rolling_forecast(lstm_model, data, target_col='pm2_5_(μg/m³)', sequence_length=24, forecast_horizon=48)

    if isinstance(prediction, pd.DataFrame):
        return {"predictions": prediction.values.flatten().tolist()}
    elif isinstance(prediction, np.ndarray):
        return {"predictions": prediction.flatten().tolist()}
    else:
        raise HTTPException(status_code=500, detail="Unsupported prediction output format.")
