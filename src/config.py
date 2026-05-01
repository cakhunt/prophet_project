from pydantic import BaseModel

class ModelConfig(BaseModel):
    yearly_seasonality: bool = True
    weekly_seasonality: bool = True
    daily_seasonality: bool = True

class TrainingConfig(BaseModel):
    test_size: float = 0.2

class AppConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()


