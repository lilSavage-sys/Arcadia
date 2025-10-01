import yfinance as yf
import pandas as pd

def fetch_stock_data(tickers, metrics):
    data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        info = stock.info
        row = {}
        for metric in metrics:
            row[metric] = info.get(metric, None)
        data[ticker] = row
    return pd.DataFrame.from_dict(data, orient='index')