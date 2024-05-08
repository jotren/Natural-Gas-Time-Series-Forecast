import pandas as pd
from sklearn.linear_model import RANSACRegressor
from tqdm import tqdm
import numpy as np

class InitialLinearRegression:
    def __init__(self, grid_locations, demand_data, weather_data_path):
        self.grid_locations = grid_locations
        self.demand_data = demand_data
        self.weather_data_path = weather_data_path
        self.models = {}  # This will now be a dictionary of dictionaries

    def preprocess_demand_data(self):
        self.demand_data['date'] = pd.to_datetime(self.demand_data['Applicable For'], dayfirst=True)
        self.demand_data['year'] = self.demand_data['date'].dt.year
        self.demand_data['day_of_year'] = self.demand_data['date'].dt.dayofyear
        self.demand_pivot = self.demand_data.pivot_table(index='day_of_year', columns='year', values='Value', aggfunc='mean')
        print("Demand Data Pivot:")
        # print(self.demand_pivot.head())

    def load_and_preprocess_weather_data(self):
        self.weather_data = {}
        for x in self.grid_locations['GRID_ID']:
            grid_level_data = pd.read_csv(f"{self.weather_data_path}/{x}.csv")
            grid_level_data['date'] = pd.to_datetime(grid_level_data['date'])
            grid_level_data['year'] = grid_level_data['date'].dt.year
            grid_level_data['day_of_year'] = grid_level_data['date'].dt.dayofyear
            temperature_data_pivot = grid_level_data.pivot_table(
                index='day_of_year', columns='year', values='airTemperature_min', aggfunc='mean')
            self.weather_data[x] = temperature_data_pivot
            print(f"Weather Data for GRID ID {x} Processed")
            # print(temperature_data_pivot.head())

    def fit_models(self):
        total_days = range(1, 366)  # Considering all days including leap years
        for day in tqdm(total_days, desc="Fitting models"):  # tqdm added here
            self.models[day] = {}
            for grid_id, weather in self.weather_data.items():
                temperature_series = weather.loc[day]
                demand_series = self.demand_pivot.loc[day]
                
                # Remove NaN values for clean data
                clean_data = pd.concat([temperature_series, demand_series], axis=1).dropna()
                if not clean_data.empty and len(clean_data) > 3:
                    X = clean_data.iloc[:, 0].values.reshape(-1, 1)  # Temperature
                    y = clean_data.iloc[:, 1].values  # Demand
                    
                    model = RANSACRegressor().fit(X, y)
                    self.models[day][grid_id] = {
                        'slope': model.estimator_.coef_[0],
                        'intercept': model.estimator_.intercept_
                    }
                else:
                    self.models[day][grid_id] = None
                    print(f"No valid data or insufficient data points for day {day}, GRID ID {grid_id}")

    def predict(self, grid_id, day_of_year, temperature):
        if day_of_year in self.models and grid_id in self.models[day_of_year]:
            model_info = self.models[day_of_year][grid_id]
            if model_info:
                return model_info['intercept'] + model_info['slope'] * temperature
            else:
                return np.nan
        else:
            return np.nan

    def save_models(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(self.models, file)
        print("Models saved successfully.")

    def load_models(self, filepath):
        with open(filepath, 'rb') as file:
            self.models = pickle.load(file)
        print("Models loaded successfully.")
