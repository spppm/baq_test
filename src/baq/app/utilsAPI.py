import pandas as pd
import joblib
import os

MODEL_PATH = "src/baq/libs/models/model.pkl"
model = None

def load_model():
    """Loads the model from disk and stores it globally."""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
    return model

def predict_pm2_5(file_content: bytes):
    """Takes CSV content, parses it into a DataFrame, and returns predictions."""
    df = pd.read_csv(pd.io.common.BytesIO(file_content))

    # Drop target column if present
    if "pm2_5" in df.columns:
        df = df.drop(columns=["pm2_5"])

    # Load model and predict
    model_instance = load_model()
    predictions = model_instance.predict(df)

    return predictions.tolist()

def predict_iterative(file_content: bytes, forecast_horizon: int = 48, sequence_length: int = 10):
    """
    Performs iterative prediction of pm2_5 for the next `forecast_horizon` steps,
    using the model that expects `sequence_length` timesteps as input.
    """
    df = pd.read_csv(pd.io.common.BytesIO(file_content))
    if "pm2_5" in df.columns:
        df = df.drop(columns=["pm2_5"])  # Drop the target if it exists

    model_instance = load_model()
    predictions = []

    # Ensure we have at least `sequence_length` rows
    if len(df) < sequence_length + forecast_horizon:
        raise ValueError(f"Input data must have at least {sequence_length + forecast_horizon} rows.")

    # Start with initial input sequence
    data = df.copy()
    input_sequence = data.iloc[:sequence_length].copy()

    for i in range(forecast_horizon):
        # Ensure model gets a proper DataFrame shape
        model_input = input_sequence.copy()
        pred = model_instance.predict(model_input)[-1]  # predict last time step
        predictions.append(pred)

        # Append predicted value into the next row to simulate actual forecast input
        if "pm2_5" not in data.columns:
            data["pm2_5"] = [None] * len(data)

        next_index = sequence_length + i
        if next_index < len(data):
            data.at[next_index, "pm2_5"] = pred

        # Update input_sequence for next round
        input_sequence = data.iloc[next_index - sequence_length + 1:next_index + 1].copy()

    return predictions