import numpy as np
import pandas as pd
from accessData import process
from scipy.optimize import fsolve, curve_fit
import OutputNumberPredictionModel as onpm

class Algorithm2: # from Alfaris's paper
    def compute_wdust(self, p_pv_percent):
        return (-0.0514 * p_pv_percent**3 + 10.2 * p_pv_percent**2 - 723 * p_pv_percent + 21300) * 1e-4

    def moving_average(self, data, window_size=60):
        """Compute the moving average with a specified window size."""
        return pd.Series(data).rolling(window=window_size, min_periods=1, center=True).mean().to_numpy()

    def detect_clouds(self, actual_power, threshold=0.1):
        """
        Detect clouds by checking for high short-term fluctuations in actual PV output.
        Returns True if clouds are likely present.
        """
        diffs = np.abs(np.diff(actual_power))
        fluctuation = np.mean(diffs)
        return fluctuation > threshold * np.max(actual_power)

    def analyze_pv_day(self, actual_power, estimated_power, cleaning_threshold=0.15):
        """
        Main function to analyze a day's PV data and decide if cleaning is needed.
        actual_power: array of measured PV output (W)
        estimated_power: array of estimated PV output for clean panel (W)
        cleaning_threshold: dust weight threshold (g/m^2) for cleaning
        Returns True if cleaning is recommended, False otherwise.
        """
        # Calculate power percentage (average over daytime hours)
        mask = estimated_power > 0.1 * np.max(estimated_power)  # focus on daytime
        p_pv_percent = 100 * np.mean(actual_power[mask]) / np.mean(estimated_power[mask])

        # Compute dust weight
        w_dust = self.compute_wdust(p_pv_percent)

        # Pattern analysis for dust (trend of moving average)
        ma = self.moving_average(actual_power[mask], window_size=60)
        x = np.arange(len(ma))
        # Fit a line to the moving average to get the slope
        slope = np.polyfit(x, ma, 1)[0]

        # Decision logic
        if (
            w_dust >= cleaning_threshold
            and slope < -0.02 * np.max(ma) / len(ma)  # negative trend
            and not self.detect_clouds(actual_power[mask])
        ):
            return True  # Cleaning needed
        else:
            return False  # No cleaning

    def main(self, data):

        cleaning_data = data.diff()
        cleaning_data = cleaning_data[cleaning_data["soiling"] > 0]
        prev_number_of_cleanings = cleaning_data.size

        scaler, original_hourly_data, hourly_data = process(data, "hourly")

        model = onpm.train(hourly_data)
        model.save('model1-h.keras')

        total_cleanings = 0
        new_ac_power = 0
        previous_ac_power = data["ac_power"].sum()
        hourly_data['day'] = hourly_data.index.dayofyear

        for i in range(0, data["ac_power"].size):
            estimated_power = []
            actual_power = []
            today = 0
            while today + i < data["ac_power"].size and (today == 0 or hourly_data['day'].iloc[today + i] == hourly_data['day'].iloc[today + i - 1]):
                original_hourly_data["soiling"].iloc[today] = 1
                hourly_data[["poa_irradiance", "ambient_temp", "wind_speed", "soiling"]].iloc[today] = scaler.transform(original_hourly_data[["poa_irradiance", "ambient_temp", "wind_speed", "soiling"]].iloc[today].to_frame().T)
                wanted_dataframe = hourly_data[["poa_irradiance", "ambient_temp", "wind_speed", "soiling"]].iloc[today].to_frame().T
                power = model.predict(pd.DataFrame(wanted_dataframe, index=wanted_dataframe.index, columns=wanted_dataframe.columns))
                estimated_power.append(power)
                actual_power.append(original_hourly_data["ac_power"].iloc[today])
                today += 1
            i += today - 1
            if self.analyze_pv_day(actual_power, estimated_power):
                total_cleanings += 1
                new_ac_power += sum(estimated_power)
            else:
                new_ac_power += sum(actual_power)
        print(f"Previous output power: {previous_ac_power}\n")
        print(f"New output power: {new_ac_power}\n")
        print(f"Previous number of cleanings: {prev_number_of_cleanings}\n")
        print(f"New number of cleanings: {total_cleanings}\n")

class Algorithm3: # From Alvarez's paper

    # Define the function corresponding to equation (19)
    # alpha # Soiling rate
    # uc  # Cleaning cost per visit ($)
    # ec # Energy revenue per kWh ($/kWh)

    # Define Equation (15) as a function of cleaning interval n
    def soiling_rate(self, data):
        df = pd.DataFrame({'t': data["Unnamed: 0"], 'eta6': data["soiling"]})

        # Define the model function
        def eta6_model(t, a, b):
            return b * np.exp(-a * t)

        # Fit the model to data
        popt, pcov = curve_fit(eta6_model, df['t'].values, df['eta6'].values, p0=[0.35, 0.99])
        a_fit, b_fit = popt
        return a_fit

    def irradiance(self, data):
        df = pd.DataFrame({'t': data["Unnamed: 0"], 'S': data["poa_irradiance"]})
        omega = 2 * np.pi / 365

        # Define the model function
        def S_model(t, a, b, theta):
            return a + b * np.cos(omega * t + theta)

        # Fit the model to data
        popt, pcov = curve_fit(S_model, df['t'].values, df['S'].values, p0=[100, 1000])
        a_fit, b_fit, theta_fit = popt
        return a_fit, b_fit

    def equation_15(self, n, alpha, a0, a1, T):
        eta1 = 0.95  # Cell efficiency, which changes with time due to degradation (0.1 - 0.25_
        eta2 = 0.98  # Cell operating temperature (0.8 - 0.9)
        eta3 = 0.98  # losses due to Joule effect on conductors as PV-transformer
        eta4 = 0.95  # Inverter efficiency
        eta5 = 0.98  # Efficiency of the maximum power point tracking (MPPT)

        eta = eta1 * eta2 * eta3 * eta4 * eta5
        omega = 2 * np.pi / 365
        n = float(n)
        # Soiling factor sum: geometric series part
        soiling_sum = (np.exp(-alpha * n) - 1) / (np.exp(-alpha) - 1)

        # Seasonal irradiance sum approximation using sinusoidal identity
        seasonal_sum = (
                T * a0 +
                (a1 / 2) * (1 + np.sin((omega / 2) * (2 * T - 1)) / np.sin(omega / 2))
        )

        # Total energy output
        ET = eta * (T / n) * soiling_sum * seasonal_sum
        return ET

    def equation_19(self, n, alpha, uc, a0, a1, T):
        n = float(n)
        ec = self.equation_15(n, alpha, a0, a1, T) * 0.1
        term1 = np.exp(alpha * n) * (np.exp(alpha) * (uc - ec) - uc)
        term2 = (n * alpha + 1) * np.exp(n) * ec
        return term1 + term2

    def main(self, data):
        # Solve the equation
        solutions = []
        a0, a1 = self.irradiance(data)
        T = 365 * (data.iloc[data.size() - 1].index.year - data.iloc[0].index.year) + data.iloc[data.size() - 1].index.dayofyear - data.iloc[0].index.dayofyear
        for uc in range (1, 29607, 10):
            optimal_n_solution = fsolve(self.equation_19, x0= np.array(100), args= (self.soiling_rate(data), uc, a0, a1, T))
            optimal_n_solution = optimal_n_solution[0]
            solutions.append(optimal_n_solution)
        np.save(solutions, 'solutions.npy')
