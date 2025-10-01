import pandas as pd
import numpy as np

def backtest_strategy(prices, signals, rebalance_freq='Q'):
    # prices: DataFrame of stock prices (date x ticker)
    # signals: DataFrame of selected stocks (date x ticker, 1 if selected, 0 otherwise)
    returns = prices.pct_change().shift(-1)
    portfolio_returns = (returns * signals).sum(axis=1) / signals.sum(axis=1)
    portfolio_returns = portfolio_returns.dropna()
    cumulative = (1 + portfolio_returns).cumprod()
    cagr = (cumulative.iloc[-1]) ** (252 / len(cumulative)) - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
    max_drawdown = (cumulative / cumulative.cummax() - 1).min()
    return {
        'cumulative': cumulative,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown
    }
