from fastapi import FastAPI, HTTPException
from more_itertools import one
from pydantic import BaseModel
import numpy as np
from keras.models import load_model
import pandas as pd
from baq.libs.utils.utilspredict import data_preprocessing, create_sequences, rolling_forecast, one_time_prediction

app = FastAPI()

modelsrc = 'src/baq/libs/models/lstm_model_3.h5'
datasrc = 'data/processed_data/data.csv'
raw_data_source = 'data/raw_data/baq_dataset.csv'

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI"}

@app.post("/predict/onetime")
async def predict_onetime():
    try:
        lstm_model = load_model(modelsrc, compile=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

    try:
        data  = pd.read_csv(datasrc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading or preprocessing data: {e}")
    
    
    prediction = one_time_prediction(lstm_model, data, target_col='pm2_5_(μg/m³)', sequence_length=24)

    return {"predictions": prediction.flatten().tolist()}

@app.post("/predict/rolling")
async def predict_rolling():
    try:
        lstm_model = load_model(modelsrc, compile=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

    try:
        data  = pd.read_csv(datasrc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading or preprocessing data: {e}")
    
    prediction = rolling_forecast(lstm_model, data, target_col='pm2_5_(μg/m³)', sequence_length=24, forecast_horizon=48)
    # Flatten nested predicted_value
    prediction["predicted_value"] = prediction["predicted_value"].apply(
        lambda x: float(x[0]) if isinstance(x, (list, np.ndarray)) else float(x)
    )

    # Convert datetime to string
    prediction["time"] = prediction["time"].apply(
        lambda x: x.isoformat() if hasattr(x, "isoformat") else str(x)
    )
    if isinstance(prediction, pd.DataFrame):
        records = prediction.to_dict(orient="records")
        return {"predictions": records}
    else:
        raise HTTPException(status_code=500, detail="Unsupported prediction output format.")

