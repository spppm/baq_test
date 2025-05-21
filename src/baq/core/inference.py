import numpy as np
import pandas as pd
from tensorflow.keras.models import Model as KerasModel

def single_step_forecasting(
    model: object,
    X_test: pd.DataFrame,
) -> pd.Series:
    """
    Generate single-step forecasts.
    
    If `model` is a tf.keras.Model (e.g. your LSTM), we reshape to 3D;
    otherwise (sklearn/XGB), we predict directly on the 2D DataFrame.
    """
    # Keras LSTM needs 3D: (samples, timesteps, features)
    if isinstance(model, KerasModel):
        # to_numpy + cast to float32, then add timestep dim
        X_np = X_test.to_numpy(dtype=np.float32).reshape(
            (len(X_test), 1, X_test.shape[1])
        )
        preds = model.predict(X_np).reshape(-1)
    else:
        preds = model.predict(X_test)

    return pd.Series(preds, index=X_test.index)

def multi_step_forecasting(
    model: object,
    X_test: pd.DataFrame,
    forecast_horizon: int,
) -> pd.Series:
    """
    Generate multi-step forecasts by iteratively predicting one step ahead,
    updating lag features, and reshaping for LSTM where needed.
    """
    X_forecast = X_test.copy()
    predictions = []

    for step in range(forecast_horizon):
        # Prepare one-row input
        X_row = X_forecast.iloc[[step]]

        if isinstance(model, KerasModel):
            # LSTM: to_numpy + float32 + reshape to (1, 1, n_features)
            X_np = X_row.to_numpy(dtype=np.float32).reshape(1, 1, X_row.shape[1])
            pred = model.predict(X_np)[0, 0]
        else:
            # sklearn or XGB
            pred = model.predict(X_row)[0]

        predictions.append(pred)

        # Update lag features for next step
        if step < forecast_horizon - 1:
            for col in X_forecast.columns:
                if col.endswith(f"_lag_{step+1}"):
                    base = col.rsplit("_lag_", 1)[0]
                    next_col = f"{base}_lag_{step+2}"
                    if next_col in X_forecast.columns:
                        X_forecast.iat[step+1, X_forecast.columns.get_loc(next_col)] = pred

    return pd.Series(predictions, index=X_test.index[:forecast_horizon])