from sklearn.preprocessing import StandardScaler

import SoilingNumberPredictionModel as snpm
import OutputNumberPredictionModel as onpm
from OutputNumberPredictionModel import log_cosh_mape, msle, smape
from accessData import accessData, delete_outliers
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

def main():
    # if len(sys.argv) not in [1, 2]:
    #     sys.exit("Usage: C:/Users\sruth\Downloads\pvdaq_system_4_2010-2016_subset_soil_signal.csv")
    path = "/Users/sruthi/Downloads/pvdaq_system_4_2010-2016_subset_soil_signal.csv"
    data = accessData(pd.read_csv(path))
    data = data[data["ac_power"] >= 0]

    data["Unnamed: 0"] = pd.to_datetime(data["Unnamed: 0"])
    days_data = data.groupby(data["Unnamed: 0"].dt.date).mean()
    days_data.index = pd.to_datetime(days_data.index)
    days_data["ac_power"] = data[["ac_power"]].groupby(data["Unnamed: 0"].dt.date).sum()
    days_data.drop("Unnamed: 0", axis=1, inplace=True)

    days_data["ac_power"] /= 1000  # units are kWatt-days
    original_days_data = days_data.copy()
    scaler = StandardScaler()
    days_data[["poa_irradiance", "ambient_temp", "wind_speed", "soiling"]] = scaler.fit_transform(days_data[["poa_irradiance", "ambient_temp", "wind_speed", "soiling"]])


    bootstrapped_df = days_data.sample(n=20000, replace=True, random_state=np.random.randint(0, 10000))
    # changed back from 10000 samples

    def generate_jittered_data(df, num_copies=5, noise_level=0.02):
        augmented_data = []
        for _ in range(num_copies):
            std_dev = df.std()
            noise = np.random.normal(loc=0, scale=noise_level * std_dev, size=df.shape)
            jittered_df = df + noise
            augmented_data.append(jittered_df)
        df_expanded = pd.concat([df] + augmented_data, ignore_index=True)
        return df_expanded

    jittered_data = generate_jittered_data(days_data, num_copies=5)
    jittered_data = delete_outliers(jittered_data)

    cleaning_data = data.diff()
    cleaning_data = cleaning_data[cleaning_data["soiling"] > 0]
    prev_number_of_cleanings = cleaning_data.size

    train_df, test_df = train_test_split(bootstrapped_df, test_size=0.3, random_state=1)

    # ac_model = onpm.FeedForwardNetwork(train_df, test_df)
    # ac_model2 = onpm.XGBoost(train_df, test_df)
    ac_model = tf.keras.models.load_model('model 2/onpm_latest5_msle.keras')

    # from pprint import pprint
    #
    # config = ac_model.get_config()
    # pprint(config)

    y_pred = ac_model.predict(days_data[["poa_irradiance", "ambient_temp", "wind_speed", "soiling"]])
    print(f"MSE: {mean_squared_error(days_data['ac_power'], y_pred)}")
    print(f"MAPE: {mean_absolute_percentage_error(days_data['ac_power'], y_pred)}")
    print(f'R^2: {r2_score(days_data["ac_power"], y_pred)}')

    # plt.scatter(np.arange(0, y_pred.size), y_pred, label = "predicted")
    # plt.scatter(np.arange(0, y_pred.size), days_data["ac_power"], label = "actual")
    # plt.legend()
    # plt.show()

    matrix = np.load("matrix6.npy")
    # plt.scatter(matrix[2], matrix[1])
    # plt.show()

    previous_ac_power = data["ac_power"].sum()
    new_ac_power = 0
    total_cleanings = 0
    days_data = original_days_data
    degradation_rate = snpm.DegredationRate(days_data)[0]/365

    cleaning_cost = 412.5 * 0.795 / 7.87432 # m^3 * dollars/m^3 for cleaning / value of electricity in dollars
    cleaning_cost_matrix = np.zeros((4, 100))
    cleaning_cost_matrix[0] = np.arange(0, 100)
    # want to store output power percent increase and cleaning percent increase at each coefficient
    execution_times = []
    previous_ac_power /= 1000

    cleaning_cost_matrix = matrix
    cleaning_cost_matrix = np.append(cleaning_cost_matrix, np.zeros((4, 9501)), axis=1)
    # cleaning_cost_matrix = np.zeros((4, 501))
    for c in range(505, 10000, 10):
        cleaning_cost_matrix[0][c] = c
        cleaning_cost = cleaning_cost_matrix[0][c]
        start_time = time.time()
        total_cleanings = 0
        new_ac_power = 0
        for today in range(0, days_data["soiling"].size - 7, 7):
            next_week = days_data.iloc[today: today + 7]
            if total_cleanings > 0:
                i = 1
                for index, day in next_week.iterrows():
                    day["soiling"] = np.maximum(0, days_data["soiling"][today - 1] + degradation_rate * i)
                    wanted_dataframe = day[["poa_irradiance", "ambient_temp", "wind_speed", "soiling"]].to_frame().T
                    day["ac_power"] = ac_model.predict(pd.DataFrame(scaler.transform(wanted_dataframe), index= wanted_dataframe.index, columns= wanted_dataframe.columns))
                    i += 1
            if next_week.size < 7:
                previous_ac_power -= next_week["ac_power"].sum()
                break
            max_p = 0
            optimal_clean_day = -1
            optimal_week = []
            for clean_day in range(0, 7):
                predicted_week = next_week.copy()
                total_p = 0
                for i in range(0, 7):
                    if i < clean_day:
                        total_p += predicted_week["ac_power"][i]
                    elif i == clean_day:
                        predicted_week["soiling"][i] = 1
                        wanted_dataframe = predicted_week[["poa_irradiance", "ambient_temp", "wind_speed", "soiling"]].iloc[i].to_frame().T
                        total_p += ac_model.predict(pd.DataFrame(scaler.transform(wanted_dataframe), index= wanted_dataframe.index, columns= wanted_dataframe.columns))
                        for day in range(i + 1, 7):
                            predicted_week["soiling"][day] = np.maximum(0, 1 + degradation_rate * (day - clean_day))
                    else:
                        wanted_dataframe = predicted_week[["poa_irradiance", "ambient_temp", "wind_speed", "soiling"]].iloc[i].to_frame().T
                        total_p += ac_model.predict(pd.DataFrame(scaler.transform(wanted_dataframe), index=wanted_dataframe.index, columns=wanted_dataframe.columns))
                if total_p > max_p:
                    max_p = total_p
                    optimal_clean_day = clean_day
                    optimal_week = predicted_week.copy()
            power_without_cleaning = next_week["ac_power"].sum()
            if optimal_clean_day != -1 and max_p > power_without_cleaning + cleaning_cost:
                total_cleanings += 1
                next_week = optimal_week
            else:
                max_p = power_without_cleaning
            new_ac_power += max_p
            days_data[today : today + 7] = next_week
        cleaning_cost_matrix[1][c] = (new_ac_power - previous_ac_power) / previous_ac_power
        cleaning_cost_matrix[2][c] = (prev_number_of_cleanings - total_cleanings) / prev_number_of_cleanings
        cleaning_cost_matrix[3][c] = cleaning_cost_matrix[2][c] / cleaning_cost_matrix[1][c]
        end_time = time.time()
        execution_times.append(end_time - start_time)
    np.save('matrix6.npy', cleaning_cost_matrix)
    np.save('execution_times2.npy', execution_times)

    max = pd.DataFrame(cleaning_cost_matrix[3]).imax()
    new_ac_power = cleaning_cost_matrix[1][max]
    total_cleanings = cleaning_cost_matrix[2][max]
    print(f"Previous output power: {previous_ac_power}\n")
    print(f"New output power: {new_ac_power}\n")
    print(f"Previous number of cleanings: {prev_number_of_cleanings}\n")
    print(f"New number of cleanings: {total_cleanings}\n")

if __name__ == "__main__":
    main()