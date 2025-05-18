import joblib
import numpy as np
import pandas as pd
from keras.models import load_model


def pm25_to_aqi_tier(pm25):
    if pm25 <= 12.0:
        return 0  # Good
    elif pm25 <= 35.4:
        return 1  # Moderate
    elif pm25 <= 55.4:
        return 2  # Unhealthy SG
    elif pm25 <= 150.4:
        return 3  # Unhealthy
    elif pm25 <= 250.4:
        return 4  # Very Unhealthy
    else:
        return 5  # Hazardous

def fill_missing_with_seasonal_median(df, seasonal_medians_csv):
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month

    stats_df = pd.read_csv(seasonal_medians_csv)

    for _, row in stats_df.iterrows():
        col = row['feature']
        condition = (
            (df['month'] == row['month']) &
            (df['day'] == row['day']) &
            (df['hour'] == row['hour']) &
            (df[col].isna())
        )
        df.loc[condition, col] = row['median']

    df.drop(columns=['hour', 'day', 'month'], inplace=True)
    return df

def clean_data(df):
    # Rename columns
    df.columns = (
        df.columns
        .str.replace('\s*\(\)', '', regex=True)
        .str.replace(' ', '_', regex=False)
    )

    # Set index
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)

    # Drop columns
    df.drop(columns=['carbon_dioxide_(ppm)', 'methane_(μg/m³)', 'snowfall_(cm)', 'snow_depth_(m)'], inplace=True)

    # Perform linear interpolation
    df = df.resample('1h').asfreq()
    df = df.interpolate(method='linear')

    # Fill missing values
    df = fill_missing_with_seasonal_median(df, 'data/raw_data/seasonal_medians.csv')

    # Encode weather code
    le = joblib.load('src/baq/libs/utils/label_encoder.pkl')
    df['weather_code_(wmo_code)'] = df['weather_code_(wmo_code)'].astype(str)
    df['weather_code_(wmo_code)'] = le.transform(df['weather_code_(wmo_code)'])

    return df

def feature_engineering(df):
    # Time-based features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_night'] = df['hour'].apply(lambda h: 1 if (h < 6 or h >= 20) else 0)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Lag features
    lags = [1, 3, 6, 12, 24]
    for lag in lags:
        for col in ['pm2_5_(μg/m³)', 'pm10_(μg/m³)', 'ozone_(μg/m³)', 'dust_(μg/m³)']:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # Rolling features
    windows = [3, 6, 12]
    for window in windows:
        for col in ['pm2_5_(μg/m³)', 'pm10_(μg/m³)', 'ozone_(μg/m³)']:
            df[f'{col}_rollmean{window}'] = df[col].rolling(window=window).mean()

    # Air quality index tier
    df['pm2_5_tier'] = df['pm2_5_(μg/m³)'].apply(pm25_to_aqi_tier)
    df.dropna(inplace=True)

    return df

def normalize_data(df, feature_cols, target_col):
    feature_scaler = joblib.load('src/baq/libs/utils/feature_scaler.pkl')
    target_scaler = joblib.load('src/baq/libs/utils/target_scaler.pkl')

    df.loc[:, feature_cols] = feature_scaler.transform(df[feature_cols])
    df.loc[:, [target_col]] = target_scaler.transform(df[[target_col]])

    return df

def create_sequences(data, target_column, sequence_length):
    X = []
    feature_data = data.drop(target_column, axis=1).values

    for i in range(len(data) - sequence_length + 1):
        X.append(feature_data[i:i + sequence_length])

    return np.array(X)

def data_preprocessing(datapath = 'data/raw_data/baq_dataset.csv'):
    df = pd.read_csv(datapath)
    df = df.head(48)
    df = clean_data(df)
    df = feature_engineering(df)

    sequence_length = 24
    feature_cols = [col for col in df.columns if col not in [
        'pm2_5_(μg/m³)', 'hour', 'dayofweek', 'month', 'is_weekend', 'is_night', 'sin_hour', 'cos_hour', 'weather_code_(wmo_code)', 'pm2_5_tier'
    ]]
    target_col = 'pm2_5_(μg/m³)'

    df = normalize_data(df, feature_cols, target_col)
    # X = create_sequences(df, target_col, sequence_length)
    # np.save('data/x.npy', X)
    df.to_csv('data/processed_data/data.csv', index=False)
    return df

def predict(model, input_data):
    y_pred = model.predict(input_data)
    return y_pred

def rolling_forecast(model, data, target_col, sequence_length, forecast_horizon):
    rolling_data = data.copy()
    rolling_forecast_df = pd.DataFrame(columns=['time', 'predicted_value'])

    for _ in range(forecast_horizon):
        input_sequence = create_sequences(rolling_data.tail(sequence_length), target_col, sequence_length)
        predicted_value = predict(model, input_sequence)

        new_row = rolling_data.tail(1)
        new_timestamp = pd.to_datetime(new_row.index[0]) + pd.Timedelta(hours=1)
        new_row.index = pd.DatetimeIndex([new_timestamp])

        rolling_data = pd.concat([rolling_data, new_row], ignore_index=True)

        rolling_forecast_df = pd.concat(
            [rolling_forecast_df, pd.DataFrame([[pd.to_datetime(rolling_data.tail(1).index), predicted_value]], columns=['time', 'predicted_value'])],
            ignore_index=True
        )

    return rolling_forecast_df


if __name__ == "__main__":
    # data_preprocessing('data/raw_data/baq_dataset.csv')


    ## Uncomment the following lines to run 1-time prediction
    # df = pd.read_csv('data/processed_data/data.csv')
    # X = create_sequences(df, target_column='pm2_5_(μg/m³)', sequence_length=24   )
    # lstm_model = load_model('src/baq/libs/models/lstm_model_3.h5', compile=False)
    # y_pred = lstm_model.predict(X)
    # y_pred = y_pred.reshape(-1, 1)
    # print(y_pred)


    # Uncomment the following lines to run rolling forecast
    df = pd.read_csv('data/processed_data/data.csv')
    lstm_model = load_model('src/baq/libs/models/lstm_model_3.h5', compile=False)
    print(rolling_forecast(lstm_model, df, target_col='pm2_5_(μg/m³)', sequence_length=24, forecast_horizon=48))

