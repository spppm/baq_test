from fastapi import FastAPI, UploadFile, File, HTTPException
from src.baq.app.utilsAPI import load_model, predict_pm2_5

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        contents = await file.read()
        predictions = predict_pm2_5(contents)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
