import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Union, Optional, List
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from baq.core.evaluation import calculate_metrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.1,
    test_size: float = 0.2
) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
    """
    Split time series data into train, validation, and test sets without shuffling.

    Args:
        X: Feature DataFrame, ordered by time.
        y: Target Series, aligned with X.
        val_size: Fraction of data for validation.
        test_size: Fraction of data for testing.

    Returns:
        Tuple of (X_train, y_train), (X_val, y_val), (X_test, y_test).
    """
    n = len(X)
    test_len = int(n * test_size)
    val_len = int(n * val_size)
    train_end = n - val_len - test_len

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:train_end + val_len], y.iloc[train_end:train_end + val_len]
    X_test, y_test = X.iloc[train_end + val_len:], y.iloc[train_end + val_len:]
    
    logger.info("Split data: train=%d, val=%d, test=%d", len(X_train), len(X_val), len(X_test))
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_lstm_model(input_shape: Tuple[int, int]) -> Model:
    """
    Build and compile an LSTM model.
    """
    inputs = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"]
    )
    logger.info("LSTM model compiled with input shape %s", input_shape)
    return model


def create_lstm_callbacks(checkpoint_path: Union[str, Path] = "best_lstm.keras") -> List:
    """
    Create callbacks for LSTM training: checkpointing, early stopping, and LR reduction.
    """
    return [
        ModelCheckpoint(
            filepath=str(checkpoint_path), save_best_only=True,
            monitor="val_loss", verbose=1
        ),
        EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.1,
            patience=5, min_lr=1e-6, verbose=1
        ),
    ]


def _train_ml_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    params: Dict,
    n_splits: int = 5,
    test_size: float = 0.2
) -> Tuple[Union[XGBRegressor, RandomForestRegressor], Dict[str, float]]:
    """
    Train and evaluate an ML model using TimeSeriesSplit CV, then retrain on full data.
    """
    # Prepare split
    n = len(X)
    test_cnt = int(n * test_size)
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_cnt)

    metrics_accumulator = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        logger.info("ML Fold %d/%d", fold, n_splits)
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        if model_name == "xgboost":
            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr)
        else:  # random_forest
            model = RandomForestRegressor(**params)
            model.fit(X_tr, y_tr)

        preds = model.predict(X_te)
        fold_metrics = calculate_metrics(y_te, preds)
        metrics_accumulator.append(fold_metrics)

    # Average metrics
    avg_metrics = {k: np.mean([m[k] for m in metrics_accumulator])
                   for k in metrics_accumulator[0].keys()}

    # Retrain final model on all data
    logger.info("Retraining %s on full dataset", model_name)
    if model_name == "xgboost":
        final_model = XGBRegressor(**params).fit(X, y)
    else:
        final_model = RandomForestRegressor(**params).fit(X, y)

    return final_model, avg_metrics


def _train_lstm_model(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.1,
    test_size: float = 0.2,
    epochs: int = 50,
    batch_size: int = 32
) -> Tuple[Model, Dict[str, float]]:
    """
    Train and evaluate an LSTM with a fixed train/val/test split,
    casting to float32 to avoid object-dtype errors.
    """
    # 1) split
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = split_train_val_test(
        X, y, val_size=val_size, test_size=test_size
    )

    # 2) ensure numeric and reshape for LSTM
    #    shape: (samples, timesteps=1, features)
    X_tr_np  = X_tr.to_numpy(dtype=np.float32).reshape((len(X_tr), 1, X_tr.shape[1]))
    X_val_np = X_val.to_numpy(dtype=np.float32).reshape((len(X_val), 1, X_val.shape[1]))
    X_te_np  = X_te.to_numpy(dtype=np.float32).reshape((len(X_te), 1, X_te.shape[1]))

    y_tr_np  = y_tr.to_numpy(dtype=np.float32)
    y_val_np = y_val.to_numpy(dtype=np.float32)
    y_te_np  = y_te.to_numpy(dtype=np.float32)

    # 3) build, compile and fit
    model     = create_lstm_model(input_shape=X_tr_np.shape[1:])
    callbacks = create_lstm_callbacks()

    model.fit(
        X_tr_np, y_tr_np,
        validation_data=(X_val_np, y_val_np),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=False,
        verbose=1
    )

    # 4) predict & evaluate
    preds = model.predict(X_te_np).reshape(-1)
    metrics = calculate_metrics(y_te_np, preds)

    return model, metrics


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    model_params: Dict,
    training_config: Dict[str, Union[int, float]]
) -> Tuple[Union[XGBRegressor, RandomForestRegressor, Model], Dict[str, float]]:
    """
    Unified training entrypoint for forecasting models.

    Dispatches to k-fold CV for ML models and train/val/test for LSTM.
    """
    model_name = model_name.lower()

    if model_name in ("xgboost", "random_forest"):
        model, metrics = _train_ml_model(
            X=X,
            y=y,
            model_name=model_name,
            params=model_params,
            n_splits=int(training_config.get("n_splits", 5)),
            test_size=float(training_config.get("test_size", 0.2))
        )
    elif model_name == "lstm":
        model, metrics = _train_lstm_model(
            X=X,
            y=y,
            val_size=float(training_config.get("val_size", 0.2)),
            test_size=float(training_config.get("test_size", 0.2)),
            epochs=int(training_config.get("epochs", 50)),
            batch_size=int(training_config.get("batch_size", 32))
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    logger.info("Completed training for %s. Metrics: %s", model_name, metrics)
    return model, metrics
