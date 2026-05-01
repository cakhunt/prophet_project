from .data_loader import load_csv, prepare_prophet_format
from .model import ProphetModel
from .config import AppConfig
from .evaluate import evaluate
from .data_loader import prepare_air_passengers
import pandas as pd
from .visualize import plot_forecast, plot_components

def train_pipeline(data_path: str):
    config = AppConfig()

    df = load_csv(data_path)
    df = prepare_air_passengers(df)

    train_df, test_df = time_series_split(df, config.training.test_size)

    model = ProphetModel(config.model)
    model.fit(train_df)

    forecast = model.predict(periods=len(test_df))

    # after prediction
    plot_forecast(model, forecast)
    plot_components(model, forecast)

    # Align predictions
    y_pred = forecast["yhat"].tail(len(test_df)).values
    y_true = test_df["y"].values

    metrics = evaluate(y_true, y_pred)

    print("Metrics:", metrics)

    return forecast

def time_series_split(df: pd.DataFrame, test_size: float):
    split_index = int(len(df) * (1 - test_size))
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    return train, test