import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def prepare_prophet_format(df: pd.DataFrame, date_col : str,target_col: str) -> pd.DataFrame:
    df = df.rename(columns={
        date_col: "ds",
        target_col: "y"
    })
    df["ds"] = pd.to_datetime[df["ds"]]
    return df

def prepare_air_passengers(df):
    df = df.rename(columns={
        "Month": "ds",
        "Passengers": "y"
    })
    df["ds"] = pd.to_datetime(df["ds"])
    return df