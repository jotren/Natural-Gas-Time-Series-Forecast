import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os
import joblib
import pandas as pd
import numpy as np

class LSTMTimeSeriesTrainer:
    def __init__(self, input_size=258, hidden_size=50, num_layers=1, dropout=0.1, weight_decay=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(self.device)
        self.linear = nn.Linear(hidden_size, 1).to(self.device)
        self.scaler = StandardScaler()
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=1000, batch_size=32, lr=0.001, verbose=True, sequence_length=10):
        self.scaler.fit(X_train)  # Fit scaler on training data
        X_train_scaled = self.scaler.transform(X_train)  # Scale training data
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)  # Scale validation data
        
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train[sequence_length-1:], dtype=torch.float32).view(-1, 1)
        
        X_train_reshaped = []
        for i in range(len(X_train_scaled) - sequence_length + 1):
            X_train_reshaped.append(X_train_tensor[i:i+sequence_length])
        X_train_reshaped = torch.stack(X_train_reshaped)
        
        assert X_train_reshaped.size(0) == y_train_tensor.size(0), "Size mismatch between X_train_reshaped and y_train_tensor"
    
        train_dataset = TensorDataset(X_train_reshaped, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=2016, shuffle=True)
    
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            [
                {'params': self.lstm.parameters()},
                {'params': self.linear.parameters()}
            ],
            lr=lr,
            weight_decay=self.weight_decay
        )
    
        for epoch in range(epochs):
            self.lstm.train()
            self.linear.train()
            train_loss = 0.0
    
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
    
                outputs, _ = self.lstm(inputs)
                outputs = self.linear(outputs[:, -1, :])
    
                # Apply dropout
                outputs = self.dropout_layer(outputs)
    
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    
                train_loss += loss.item() * inputs.size(0)
    
            train_loss /= len(train_loader.dataset)
    
            if verbose and (epoch + 1) % 500 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}')


    def evaluate(self, X_val, y_val, criterion):
        self.lstm.eval()
        self.linear.eval()
        with torch.no_grad():
            X_val_scaled = torch.tensor(self.scaler.transform(X_val.reshape(-1, 1)), dtype=torch.float32).unsqueeze(1).to(self.device)
            y_val = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32).to(self.device)
            outputs, _ = self.lstm(X_val_scaled)
            outputs = self.linear(outputs[:, -1, :])
            val_loss = criterion(outputs, y_val)
        return val_loss.item()
        
    def predict(self, X_test, decimal_places=7):
        self.lstm.eval()
        self.linear.eval()
        
        # Scale and reshape X_test
        X_test_scaled = self.scaler.transform(X_test)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        X_test_reshaped = X_test_tensor.unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            outputs, _ = self.lstm(X_test_reshaped)
            predictions = self.linear(outputs[:, -1, :])
        
        # Move predictions to CPU and convert to NumPy array
        predictions_cpu = predictions.cpu().numpy()
        
        # Round predictions to the specified number of decimal places
        rounded_predictions = np.round(predictions_cpu, decimals=decimal_places)
        
        # Flatten the array of arrays to get a simple array of numbers
        flattened_predictions = rounded_predictions.flatten()
        
        return flattened_predictions

    def save_model(self, directory="models"):
        os.makedirs(os.path.join(directory, "scalers"), exist_ok=True)
        scaler_filename = os.path.join(directory, "scalers", f"{self.__class__.__name__}_scaler.joblib")
        joblib.dump(self.scaler, scaler_filename)
        print(f"Scaler saved to {scaler_filename}")
        
        model_filename = os.path.join(directory, "trained-models", f"{self.__class__.__name__}_model.pth")
        torch.save({
            'lstm_state_dict': self.lstm.state_dict(),
            'linear_state_dict': self.linear.state_dict()
        }, model_filename)
        print(f"Model saved to {model_filename}")

    def load_model(self, directory="../models"):
        scaler_filename = os.path.join(directory, "scalers", f"{self.__class__.__name__}_scaler.joblib")
        self.scaler = joblib.load(scaler_filename)
        print("Scaler loaded.")

        model_filename = os.path.join(directory, "trained-models", f"{self.__class__.__name__}_model.pth")
        checkpoint = torch.load(model_filename, map_location=self.device)
        self.lstm.load_state_dict(checkpoint['lstm_state_dict'])
        self.linear.load_state_dict(checkpoint['linear_state_dict'])
        print("Model loaded.")

    def predict_df(self, df, feature_names):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if not all(name in df for name in feature_names):
            raise ValueError("DataFrame must contain all required feature columns.")
        df_features = df[feature_names]
        scaled_features = self.scaler.transform(df_features.values.reshape(-1, 1))
        scaled_features = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(1).to(self.device)
        with torch.no_grad():
            outputs, _ = self.lstm(scaled_features)
            prediction = self.linear(outputs[:, -1, :]).item()
        return prediction
