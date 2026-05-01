from prophet import Prophet
import pandas as pd
from .config import ModelConfig


class ProphetModel:
    def __init__(self, config: ModelConfig):
        self.model = Prophet(
            yearly_seasonality=config.yearly_seasonality,
            weekly_seasonality=config.weekly_seasonality,
            daily_seasonality=config.daily_seasonality,
        )

    def fit(self, df: pd.DataFrame):
        self.model.fit(df)

    def predict(self, periods: int, freq: str = "D") -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        print("dummy print")
        return forecast
