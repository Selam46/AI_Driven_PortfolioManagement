# AI-Driven Portfolio Management

This project implements an AI-driven approach to portfolio management, focusing on time series analysis, forecasting, and portfolio optimization. The project combines traditional financial analysis with modern machine learning techniques to create an intelligent portfolio management system.

## Project Structure

```
AI_Driven_PortfolioManagement/
├── data/                  # Stock price data and other datasets
├── notebook/             # Jupyter notebooks for analysis
│   ├── time_series_forecasting.ipynb
│   ├── market_forecasting.ipynb
│   └── portfolio_optimization.ipynb
├── src/                  # Source code
│   ├── model_utils.py    # Time series models implementation
│   ├── market_forecasting.py
│   └── portfolio_optimization.py
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Features

1. **Time Series Analysis**
   - Data preprocessing and feature engineering
   - Technical indicator calculation
   - Statistical analysis of stock price movements

2. **Market Forecasting**
   - Implementation of multiple forecasting models:
     - ARIMA/SARIMA
     - Prophet
     - LSTM Neural Networks
   - Model comparison and evaluation
   - Confidence interval estimation

3. **Portfolio Optimization**
   - Multi-asset portfolio management
   - Risk-return optimization
   - Sharpe ratio maximization
   - Value at Risk (VaR) calculation
   - Efficient frontier visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Selam46/AI_Driven_PortfolioManagement.git
cd AI_Driven_PortfolioManagement
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Data Collection**
   - Historical data is automatically downloaded using the yfinance library
   - Currently supports TSLA, BND, and SPY

2. **Time Series Analysis**
   - Run the time_series_forecasting.ipynb notebook for initial analysis

3. **Market Forecasting**
   - Use market_forecasting.ipynb to generate future price predictions
   - Compare different model performances

4. **Portfolio Optimization**
   - Execute portfolio_optimization.ipynb for portfolio analysis
   - View optimal allocations and performance metrics

## Dependencies

- Python 3.8+
- pandas
- numpy
- scipy
- yfinance
- scikit-learn
- tensorflow
- prophet
- matplotlib
- seaborn

## Results

The project provides:
- Future price predictions with confidence intervals
- Optimal portfolio allocations
- Risk-return analysis
- Performance visualizations
- Investment recommendations


