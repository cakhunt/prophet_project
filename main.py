from src.train import train_pipeline

def train():
    data_path = 'D:\\WORKSPACE\\time_series\\prophet_project\\data\\raw\\air_passengers.csv'
    forecast = train_pipeline(data_path)
    print(forecast.tail())

if __name__ == "__main__":
    train()