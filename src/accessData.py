import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler

def accessData(path):
    data = pd.read_csv(path)
    data = data[["Unnamed: 0", "ac_power", "poa_irradiance", "ambient_temp", "wind_speed", "soiling"]]
    for col in ["ac_power", "poa_irradiance", "ambient_temp", "wind_speed", "soiling"]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data.dropna(inplace=True)

    # Code to delete outliers
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
        scale = StandardScaler()
        data = scale.fit_transform(data.iloc[:, 2:])
        data = pd.DataFrame(data.iloc[:, 2:], columns=data.columns[2:])

    return data