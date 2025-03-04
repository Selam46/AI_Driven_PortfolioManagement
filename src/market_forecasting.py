import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta

class MarketForecaster:
    def __init__(self, model=None, data=None):
        self.model = model
        self.data = data
        self.forecast_results = None
        self.confidence_intervals = None
        
    def generate_forecast(self, steps=180, alpha=0.05):
        """
        Generate forecasts using the trained model
        
        Parameters:
        -----------
        steps : int
            Number of steps to forecast (default 180 days / 6 months)
        alpha : float
            Significance level for confidence intervals
            
        Returns:
        --------
        forecast_results : pd.Series
            Point forecasts
        confidence_intervals : tuple
            Lower and upper confidence bounds
        """
        if self.model is None:
            raise ValueError("Model not initialized. Please train or load a model first.")
            
        # Generate forecast
        forecast = self.model.get_forecast(steps=steps)
        
        # Get confidence intervals
        try:
            conf_int = forecast.conf_int(alpha=alpha)
        except (AttributeError, NotImplementedError):
            # If confidence intervals are not available, create dummy intervals
            forecast_values = forecast if isinstance(forecast, pd.Series) else pd.Series(forecast)
            std_dev = forecast_values.std()
            lower = forecast_values - 2 * std_dev
            upper = forecast_values + 2 * std_dev
            conf_int = pd.DataFrame({'lower': lower, 'upper': upper})
        
        # Store results
        self.forecast_results = forecast.predicted_mean if hasattr(forecast, 'predicted_mean') else forecast
        self.confidence_intervals = conf_int
        
        return self.forecast_results, self.confidence_intervals
    
    def plot_forecast(self, title="Tesla Stock Price Forecast", figsize=(15, 8)):
        """
        Plot the historical data, forecasts, and confidence intervals
        """
        plt.figure(figsize=figsize)
        
        # Plot historical data
        plt.plot(self.data.index, self.data, label='Historical Data', color='blue')
        
        # Plot forecast
        forecast_index = pd.date_range(
            start=self.data.index[-1] + timedelta(days=1),
            periods=len(self.forecast_results),
            freq='D'
        )
        plt.plot(forecast_index, self.forecast_results, label='Forecast', color='red')
        
        # Plot confidence intervals
        plt.fill_between(
            forecast_index,
            self.confidence_intervals.iloc[:, 0],
            self.confidence_intervals.iloc[:, 1],
            color='red',
            alpha=0.2,
            label='95% Confidence Interval'
        )
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid(True)
        
        return plt
    
    def analyze_trends(self):
        """
        Analyze trends in the forecast data
        
        Returns:
        --------
        dict : Dictionary containing trend analysis results
        """
        if self.forecast_results is None:
            raise ValueError("No forecast results available. Please generate forecast first.")
            
        # Convert forecast_results to pandas Series if it's not already
        if not isinstance(self.forecast_results, pd.Series):
            forecast_series = pd.Series(self.forecast_results)
        else:
            forecast_series = self.forecast_results
            
        # Calculate overall trend
        overall_trend = 'Upward' if forecast_series.iloc[-1] > forecast_series.iloc[0] else 'Downward'
        
        # Calculate volatility
        forecast_volatility = forecast_series.std()
        
        # Calculate confidence interval width if available
        if self.confidence_intervals is not None:
            ci_width = (self.confidence_intervals.iloc[:, 1] - self.confidence_intervals.iloc[:, 0]).mean()
        else:
            ci_width = 2 * forecast_volatility  # Use 2 standard deviations as approximate CI width
        
        # Calculate expected return
        expected_return = ((forecast_series.iloc[-1] - forecast_series.iloc[0]) / forecast_series.iloc[0]) * 100
        
        return {
            'overall_trend': overall_trend,
            'volatility': forecast_volatility,
            'confidence_interval_width': ci_width,
            'expected_return_percent': expected_return
        }
    
    def generate_market_insights(self):
        """
        Generate market insights based on the forecast
        
        Returns:
        --------
        dict : Dictionary containing market insights
        """
        if self.forecast_results is None:
            raise ValueError("No forecast results available. Please generate forecast first.")
            
        analysis = self.analyze_trends()
        
        # Convert forecast_results to pandas Series if it's not already
        if not isinstance(self.forecast_results, pd.Series):
            forecast_series = pd.Series(self.forecast_results)
        else:
            forecast_series = self.forecast_results
        
        insights = {
            'trend_direction': analysis['overall_trend'],
            'risk_level': 'High' if analysis['volatility'] > self.data.std() else 'Moderate',
            'opportunity_score': analysis['expected_return_percent'] / analysis['volatility'],
            'confidence_level': 'Low' if analysis['confidence_interval_width'] > 2 * forecast_series.std() else 'High'
        }
        
        return insights 