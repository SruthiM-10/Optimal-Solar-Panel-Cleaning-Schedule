from sklearn.preprocessing import StandardScaler

import SoilingNumberPredictionModel as snpm
import OutputNumberPredictionModel as onpm
from accessData import accessData, process
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
import time
import matplotlib.pyplot as plt
from AlternateStrategies import Algorithm2

def main():
    path = "/Users/sruthi/Downloads/pvdaq_system_4_2010-2016_subset_soil_signal.csv"
    data = accessData(pd.read_csv(path))

    # from pprint import pprint
    #
    # config = ac_model.get_config()
    # pprint(config)

    algorithm1(data)

def algorithm1(data):
    cleaning_data = data.diff()
    cleaning_data = cleaning_data[cleaning_data["soiling"] > 0]
    prev_number_of_cleanings = cleaning_data.size

    # matrix = np.load("matrix6.npy")
    scaler, original_days_data, days_data = process(data, "daily")
    # model = onpm.train(days_data)
    # model.save('model12.keras')
    model = tf.keras.models.load_model('/Users/sruthi/PycharmProjects/pvCleaningProject2025/src/model12.keras')

    previous_ac_power = data["ac_power"].sum()
    days_data = original_days_data
    degradation_rate = snpm.DegredationRate(days_data)[0]/365

    cleaning_cost = 412.5 * 0.795 / 7.87432 # m^3 * dollars/m^3 for cleaning / value of electricity in dollars
    cleaning_cost_matrix = np.zeros((4, 253779))
    cleaning_cost_matrix[0] = np.arange(0, 253779)
    # want to store output power percent increase and cleaning percent increase at each coefficient
    execution_times = []
    previous_ac_power /= 1000

    # cleaning_cost_matrix = matrix
    last_c = -1
    for c in range(0, 253779, 5):
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
                    day["soiling"] = np.maximum(0, days_data["soiling"].iloc[today - 1] + degradation_rate * i)
                    wanted_dataframe = day[["poa_irradiance", "ambient_temp", "wind_speed", "soiling", "day", "month"]].to_frame().T
                    day["ac_power"] = model.predict(pd.DataFrame(scaler.transform(wanted_dataframe), index= wanted_dataframe.index, columns= wanted_dataframe.columns))
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
                        total_p += predicted_week["ac_power"].iloc[i]
                    elif i == clean_day:
                        predicted_week.loc[predicted_week.index[i],"soiling"] = 1
                        wanted_dataframe = predicted_week[["poa_irradiance", "ambient_temp", "wind_speed", "soiling", "day", "month"]].iloc[i].to_frame().T
                        total_p += model.predict(pd.DataFrame(scaler.transform(wanted_dataframe), index= wanted_dataframe.index, columns= wanted_dataframe.columns))
                        for day in range(i + 1, 7):
                            predicted_week.loc[predicted_week.index[day],"soiling"] = np.maximum(0, 1 + degradation_rate * (day - clean_day))
                    else:
                        wanted_dataframe = predicted_week[["poa_irradiance", "ambient_temp", "wind_speed", "soiling", "day", "month"]].iloc[i].to_frame().T
                        total_p += model.predict(pd.DataFrame(scaler.transform(wanted_dataframe), index=wanted_dataframe.index, columns=wanted_dataframe.columns))
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
        if total_cleanings == 0:
            last_c = c
            break
    np.save('matrix7.npy', cleaning_cost_matrix)
    np.save('execution_times3.npy', execution_times)
    print(f"Greatest cleaning cost: {last_c}\n")

    max = pd.DataFrame(cleaning_cost_matrix[3]).imax()
    new_ac_power = cleaning_cost_matrix[1][max]
    total_cleanings = cleaning_cost_matrix[2][max]
    print(f"Previous output power: {previous_ac_power}\n")
    print(f"New output power: {new_ac_power}\n")
    print(f"Previous number of cleanings: {prev_number_of_cleanings}\n")
    print(f"New number of cleanings: {total_cleanings}\n")

def get_prediction_intervals(model, X_new, n_samples=1000, alpha=0.05):
    y_pred = model.predict(X_new[["poa_irradiance", "ambient_temp", "wind_speed", "soiling", "day", "month"]]).flatten()
    residuals = X_new["ac_power"] - y_pred

    all_preds = np.zeros((n_samples, len(X_new)))
    for i in range(n_samples):
        sampled_residuals = np.random.choice(residuals, size=len(X_new), replace=True)
        all_preds[i] = y_pred + sampled_residuals

    lower = np.percentile(all_preds, 100 * alpha / 2, axis=0)
    upper = np.percentile(all_preds, 100 * (1 - alpha / 2), axis=0)
    return lower, upper

if __name__ == "__main__":
    main()

'''
TO DO:
Maybe later repeat for two more datasets?
'''