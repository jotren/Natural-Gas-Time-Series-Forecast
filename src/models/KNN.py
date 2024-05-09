from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import os
import joblib
import pandas as pd

class KNNTimeSeriesTrainer:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        print("Model trained.")

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        return predictions

    def save_model(self, directory="models"):
        os.makedirs(os.path.join(directory, "trained-models"), exist_ok=True)
        os.makedirs(os.path.join(directory, "scalers"), exist_ok=True)
        
        model_filename = f"{directory}/trained-models/{self.__class__.__name__}_model.joblib"
        scaler_filename = f"{directory}/scalers/{self.__class__.__name__}_scaler.joblib"
        
        joblib.dump(self.model, model_filename)
        joblib.dump(self.scaler, scaler_filename)
        print(f"Model saved to {model_filename}")
        print(f"Scaler saved to {scaler_filename}")
        
    def load_model(self, directory="models"):
        model_filename = f"{directory}/trained-models/{self.__class__.__name__}_model.joblib"
        scaler_filename = f"{directory}/scalers/{self.__class__.__name__}_scaler.joblib"
        
        self.model = joblib.load(model_filename)
        self.scaler = joblib.load(scaler_filename)
        print("Model and scaler loaded.")

    def predict_df(self, df, feature_names):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if not all(name in df for name in feature_names):
            raise ValueError("DataFrame must contain all required feature columns.")
        df_features = df[feature_names]
        scaled_features = self.scaler.transform(df_features)
        predictions = self.model.predict(scaled_features)
        return predictions
