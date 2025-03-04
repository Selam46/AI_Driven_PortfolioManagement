import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Dict, Tuple

class DataProcessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and ensuring correct data types
        """
        # Handle missing values
        self.data = self.data.fillna(method='ffill')
        
        # Ensure datetime index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
            
        return self.data
    
    def get_basic_stats(self) -> Dict:
        """
        Calculate basic statistics of the data
        """
        stats = {
            'summary': self.data.describe(),
            'missing_values': self.data.isnull().sum(),
            'data_types': self.data.dtypes
        }
        return stats
    
    def detect_outliers(self, column: str = 'Close', threshold: float = 3) -> pd.Series:
        """
        Detect outliers using z-score method
        """
        z_scores = np.abs((self.data[column] - self.data[column].mean()) / self.data[column].std())
        return self.data[column][z_scores > threshold]
    
    def decompose_time_series(self, column: str = 'Close', period: int = 252) -> Dict:
        """
        Decompose time series into trend, seasonal, and residual components
        """
        decomposition = seasonal_decompose(self.data[column], period=period)
        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        }
