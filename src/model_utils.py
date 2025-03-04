import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def calculate_metrics(y_true: np.array, y_pred: np.array) -> Dict[str, float]:
    """
    Calculate performance metrics for time series predictions
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

def train_test_split_ts(data: pd.Series, train_size: float = 0.8) -> Tuple[pd.Series, pd.Series]:
    """
    Split time series data into training and testing sets
    """
    n = len(data)
    train_size = int(n * train_size)
    return data[:train_size], data[train_size:]

def create_sequences(data: np.array, seq_length: int) -> Tuple[np.array, np.array]:
    """
    Create sequences for neural network training
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

class ARIMAModel:
    def __init__(self):
        self.model = None
        
    def grid_search(self, data: pd.Series) -> tuple:
        """
        Grid search for ARIMA parameters
        """
        best_aic = float('inf')
        best_order = None
        
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        model = ARIMA(data, order=(p, d, q)).fit()
                        if model.aic < best_aic:
                            best_aic = model.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        return best_order
    
    def fit(self, data: pd.Series, order: tuple = None) -> None:
        """
        Fit ARIMA model
        """
        if order is None:
            order = self.grid_search(data)
        self.model = ARIMA(data, order=order).fit()
        self.order = order
        
    def predict(self, steps: int) -> pd.Series:
        """
        Generate predictions
        """
        return self.model.forecast(steps)

class SimpleNeuralNetwork:
    def __init__(self, seq_length: int = 10):
        self.seq_length = seq_length
        self.model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=1000
        )
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, data: pd.Series) -> Tuple[np.array, np.array]:
        """
        Prepare data for neural network
        """
        # Scale the data
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        # Create sequences
        X, y = create_sequences(scaled_data.flatten(), self.seq_length)
        return X, y
    
    def fit(self, data: pd.Series) -> None:
        """
        Fit neural network model
        """
        X, y = self.prepare_data(data)
        self.model.fit(X, y.ravel())
        
    def predict(self, data: pd.Series, steps: int) -> np.array:
        """
        Generate predictions
        """
        # Prepare the last sequence from data
        scaled_data = self.scaler.transform(data.values.reshape(-1, 1))
        last_sequence = scaled_data[-self.seq_length:].flatten()
        
        predictions = []
        for _ in range(steps):
            # Predict next value
            next_pred = self.model.predict(last_sequence.reshape(1, -1))
            predictions.append(next_pred[0])
            
            # Update sequence
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[-1] = next_pred
            
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        return self.scaler.inverse_transform(predictions).flatten()

class SARIMAModel:
    def __init__(self):
        self.model = None
        
    def grid_search(self, data: pd.Series) -> Tuple[tuple, tuple]:
        """
        Grid search for SARIMA parameters
        """
        best_aic = float('inf')
        best_order = None
        best_seasonal_order = None
        
        # Simplified grid search
        orders = [(1,1,1), (1,1,2), (2,1,1), (2,1,2)]
        seasonal_orders = [(1,1,1,12), (1,1,0,12), (0,1,1,12)]
        
        for order in orders:
            for seasonal_order in seasonal_orders:
                try:
                    model = SARIMAX(
                        data,
                        order=order,
                        seasonal_order=seasonal_order
                    ).fit(disp=False)
                    
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_order = order
                        best_seasonal_order = seasonal_order
                except:
                    continue
        
        return best_order, best_seasonal_order
    
    def fit(self, data: pd.Series, order: tuple = None, seasonal_order: tuple = None) -> None:
        """
        Fit SARIMA model
        """
        if order is None or seasonal_order is None:
            order, seasonal_order = self.grid_search(data)
            
        self.model = SARIMAX(
            data,
            order=order,
            seasonal_order=seasonal_order
        ).fit(disp=False)
        
        self.order = order
        self.seasonal_order = seasonal_order
        
    def predict(self, steps: int) -> pd.Series:
        """
        Generate predictions
        """
        return self.model.forecast(steps)
    
    def get_fitted_values(self) -> pd.Series:
        """
        Get in-sample predictions
        """
        return self.model.get_prediction().predicted_mean

class ProphetModel:
    def __init__(self):
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
    def prepare_data(self, data: pd.Series) -> pd.DataFrame:
        """
        Prepare data for Prophet format
        """
        # Convert index to datetime and remove timezone
        df = pd.DataFrame({
            'ds': data.index.tz_localize(None),  # Remove timezone
            'y': data.values
        })
        return df
    
    def fit(self, data: pd.Series) -> None:
        """
        Fit Prophet model
        """
        train_df = self.prepare_data(data)
        self.model.fit(train_df)
        
    def predict(self, periods: int) -> pd.DataFrame:
        """
        Generate predictions
        """
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        return forecast

class LSTMModel:
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        
    def create_sequences(self, data: np.array) -> Tuple[np.array, np.array]:
        """
        Create sequences for LSTM training
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: tuple) -> None:
        """
        Build LSTM model architecture
        """
        self.model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        self.model.compile(optimizer=Adam(), loss='mse')
        
    def prepare_data(self, data: pd.Series) -> Tuple[np.array, np.array]:
        """
        Prepare data for LSTM
        """
        # Scale the data
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        # Reshape for LSTM [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def fit(self, data: pd.Series, epochs: int = 50, batch_size: int = 32) -> None:
        """
        Fit LSTM model
        """
        X, y = self.prepare_data(data)
        
        if self.model is None:
            self.build_model(input_shape=(X.shape[1], 1))
        
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
    
    def predict(self, data: pd.Series, steps: int) -> np.array:
        """
        Generate predictions
        """
        # Prepare the last sequence from data
        scaled_data = self.scaler.transform(data.values.reshape(-1, 1))
        last_sequence = scaled_data[-self.sequence_length:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            # Reshape for prediction
            current_sequence_reshaped = current_sequence.reshape((1, self.sequence_length, 1))
            
            # Predict next value
            next_pred = self.model.predict(current_sequence_reshaped, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred
            
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        return self.scaler.inverse_transform(predictions).flatten()
