import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Tuple, List
import yfinance as yf

class PortfolioOptimizer:
    def __init__(self, data: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize portfolio optimizer
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with daily prices for each asset
        risk_free_rate : float
            Annual risk-free rate (default 2%)
        """
        self.data = data
        self.returns = data.pct_change().dropna()
        self.risk_free_rate = risk_free_rate
        self.daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1
        
    def calculate_portfolio_metrics(self, weights: np.array) -> Dict[str, float]:
        """
        Calculate portfolio metrics including return, volatility, and Sharpe ratio
        """
        # Convert weights to array if needed
        weights = np.array(weights)
        
        # Calculate portfolio return
        returns = np.sum(self.returns.mean() * weights) * 252
        
        # Calculate portfolio volatility
        volatility = np.sqrt(
            np.dot(weights.T, np.dot(self.returns.cov() * 252, weights))
        )
        
        # Calculate Sharpe ratio
        sharpe_ratio = (returns - self.risk_free_rate) / volatility
        
        return {
            'return': returns,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    def calculate_asset_metrics(self) -> pd.DataFrame:
        """
        Calculate individual asset metrics
        """
        annual_returns = self.returns.mean() * 252
        annual_volatility = self.returns.std() * np.sqrt(252)
        sharpe_ratios = (annual_returns - self.risk_free_rate) / annual_volatility
        
        metrics = pd.DataFrame({
            'Annual Return': annual_returns,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratios
        })
        
        return metrics
    
    def calculate_var(self, confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate Value at Risk for each asset
        """
        var_dict = {}
        for column in self.returns.columns:
            returns = self.returns[column].values
            var = np.percentile(returns, (1 - confidence_level) * 100)
            var_dict[column] = -var * self.data[column].iloc[-1]
            
        return var_dict
    
    def optimize_portfolio(self, objective: str = 'sharpe') -> Tuple[np.array, Dict[str, float]]:
        """
        Optimize portfolio weights based on objective
        
        Parameters:
        -----------
        objective : str
            'sharpe' for maximum Sharpe ratio
            'return' for maximum return
            'risk' for minimum risk
        """
        num_assets = len(self.data.columns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # weights sum to 1
        bounds = tuple((0, 1) for _ in range(num_assets))  # weights between 0 and 1
        
        if objective == 'sharpe':
            def objective_function(weights):
                return -self.calculate_portfolio_metrics(weights)['sharpe_ratio']
        elif objective == 'return':
            def objective_function(weights):
                return -self.calculate_portfolio_metrics(weights)['return']
        else:  # minimize risk
            def objective_function(weights):
                return self.calculate_portfolio_metrics(weights)['volatility']
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/num_assets] * num_assets)
        
        # Optimize
        result = minimize(
            objective_function,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        metrics = self.calculate_portfolio_metrics(optimal_weights)
        
        return optimal_weights, metrics
    
    def calculate_cumulative_returns(self, weights: np.array) -> pd.Series:
        """
        Calculate cumulative returns for the portfolio
        """
        portfolio_returns = np.sum(self.returns * weights, axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        return cumulative_returns
    
    @staticmethod
    def download_data(symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download historical data for multiple symbols
        """
        data = pd.DataFrame()
        
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            data[symbol] = hist['Close']
            
        return data 