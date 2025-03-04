import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import yfinance as yf
from datetime import datetime

def fetch_stock_data(tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple tickers from Yahoo Finance
    """
    data_dict = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        data_dict[ticker] = stock.history(start=start_date, end=end_date)
    return data_dict

def calculate_daily_returns(df: pd.DataFrame) -> pd.Series:
    """
    Calculate daily returns from adjusted close prices
    """
    return df['Close'].pct_change()

def calculate_rolling_metrics(df: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate rolling mean and standard deviation
    """
    returns = calculate_daily_returns(df)
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    return rolling_mean, rolling_std

def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR)
    """
    return np.percentile(returns.dropna(), (1 - confidence_level) * 100)

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe Ratio
    """
    excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
    return np.sqrt(252) * excess_returns.mean() / returns.std()
