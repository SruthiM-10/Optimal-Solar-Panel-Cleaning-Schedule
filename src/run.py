import SoilingNumberPredictionModel as snpm
import OutputNumberPredictionModel as onpm
from accessData import accessData
import sys
import pandas as pd

def main():
    # if len(sys.argv) not in [1, 2]:
    #     sys.exit("Usage: C:/Users\sruth\Downloads\pvdaq_system_4_2010-2016_subset_soil_signal.csv")
    path = "/Users/sruthi/Downloads/pvdaq_system_4_2010-2016_subset_soil_signal.csv"
    data = accessData(pd.read_csv(path))

    data["Unnamed: 0"] = pd.to_datetime(data["Unnamed: 0"])
    days_data = data.groupby(pd.Grouper(key='Unnamed: 0', freq='D'), sort=False)[['poa_irradiance', 'ambient_temp', 'wind_speed', 'soiling']].mean()
    days_data["ac_power"] = data.groupby(pd.Grouper(key='date', freq='D'), sort= False)['ac_power'].sum()["ac_power"] # units are watt-minutes

    cleaning_data = data.diff()
    cleaning_data = cleaning_data[cleaning_data["soiling"] > 0]
    prev_number_of_cleanings = cleaning_data.size

    ac_model = onpm.NeuralNetwork(days_data)

    previous_ac_power = data["ac_power"].sum()
    new_ac_power = 0
    total_cleanings = 0
    degradation_rate = snpm.DegredationRate(days_data)

    for row in days_data:
        today = row.index
        next_week = days_data.iloc[today: today + 7]
        soiling_init = next_week["soiling"][0]
        if total_cleanings > 0:
            for i in range(today, today + 7):
                next_week["soiling"][i] = soiling_init - degradation_rate (i - today) # or change the soiling model to incorporate values from the previous week
        max_p = 0
        cleanToday = False
        for clean_day in range(today, today + 8):
            total_p = 0
            for i in range(today, today + 7):
                if i != clean_day:
                    total_p += ac_model.predict(next_week["poa_irradiance", "ambient_temp", "wind_speed", "soiling"][i])
                else:
                    next_week["soiling"][i] = 1
                    total_p += ac_model.predict(next_week["poa_irradiance", "ambient_temp", "wind_speed"][i])
                    for i in range(i, today + 7):
                        next_week["soiling"][i] = 1 - degradation_rate(i - today)
            if total_p > max_p:
                max_p = total_p
                if clean_day == today:
                    cleanToday = True
        if cleanToday:
            total_cleanings += 1
        new_ac_power += max_p

    print(f"Previous output power: {previous_ac_power}\n")
    print(f"New output power: {new_ac_power}\n")
    print(f"Previous number of cleanings: {prev_number_of_cleanings}\n")
    print(f"New number of cleanings: {total_cleanings}\n")

if __name__ == "__main__":
    main()