import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def accessData(data):
    data = data[["Unnamed: 0", "ac_power", "poa_irradiance", "ambient_temp", "wind_speed", "soiling"]]
    for col in ["ac_power", "poa_irradiance", "ambient_temp", "wind_speed", "soiling"]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data.dropna(inplace=True)
    delete_outliers(data)
    data = data[data["ac_power"] >= 0]
    data["Unnamed: 0"] = pd.to_datetime(data["Unnamed: 0"])
    data.set_index("Unnamed: 0", inplace=True)
    return data

def delete_outliers(data):
    for col in ["ac_power", "poa_irradiance", "ambient_temp", "wind_speed", "soiling"]:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1

        lb = q1 - (iqr * 1.5)
        ub = q3 + (iqr * 1.5)

        def apply(x):
            if x > ub:
                return ub
            elif x < lb:
                return lb
            else:
                return x

        data[col] = data[col].apply(apply)
    return data

def process(data, type):
    if type == "daily":
        days_data = data.groupby(data["Unnamed: 0"].dt.date).mean()
        days_data.index = pd.to_datetime(days_data.index)
        days_data["ac_power"] = data[["ac_power"]].groupby(data["Unnamed: 0"].dt.date).sum()
        days_data["ac_power"] /= 1000  # units are kWatt-days
        original_days_data = days_data.copy()
        scaler = StandardScaler()
        days_data[["poa_irradiance", "ambient_temp", "wind_speed", "soiling"]] = scaler.fit_transform(
            days_data[["poa_irradiance", "ambient_temp", "wind_speed", "soiling"]])
        seed = 42
        np.random.seed(seed)
        bootstrapped_df = days_data.sample(n=20000, replace=True, random_state=seed)
        bootstrapped_df.sort_index(inplace= True)
        return scaler, original_days_data, days_data
    elif type == "hourly":
        data.index = pd.to_datetime(data.index)
        hourly_data = data[["ac_power", "poa_irradiance", "ambient_temp", "wind_speed", "soiling"]].resample('h').mean()
        hourly_data.index.name = None
        hourly_data["ac_power"] = data[["ac_power"]].resample('h').sum()
        hourly_data.dropna(inplace=True)
        hourly_data["ac_power"] /= 1000  # units are kWatt-hours
        original_hourly_data = hourly_data.copy()
        scaler = StandardScaler()
        hourly_data[["poa_irradiance", "ambient_temp", "wind_speed", "soiling"]] = scaler.fit_transform(
            hourly_data[["poa_irradiance", "ambient_temp", "wind_speed", "soiling"]])
        return scaler, original_hourly_data, hourly_data