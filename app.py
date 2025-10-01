import scipy.optimize

# Rolling Beta Tab (move after sidebar_tabs and portfolio are defined)
import scipy.optimize

def backtest_equity_curve(prices, weights):
    # Simple weighted portfolio equity curve
    returns = prices.pct_change().fillna(0)
    portfolio_ret = (returns * weights).sum(axis=1)
    equity_curve = (1 + portfolio_ret).cumprod()

    return equity_curve, portfolio_ret

def sharpe_ratio(returns):
    return returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan

def sortino_ratio(returns, target=0):
    downside = returns[returns < target]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else np.nan
    return (returns.mean() - target) / downside_std if downside_std > 0 else np.nan

def information_ratio(returns, benchmark_returns):
    diff = returns - benchmark_returns
    tracking_error = diff.std() * np.sqrt(252) if diff.std() > 0 else np.nan
    return diff.mean() / tracking_error if tracking_error > 0 else np.nan

def calmar_ratio(returns, equity_curve):
    # max_drawdown removed
    max_dd = np.nan
    annual_return = returns.mean() * 252
    return np.nan

def treynor_ratio(returns, beta):
    rf = 0.01 # risk-free rate
    excess_return = returns.mean() * 252 - rf
    return excess_return / beta if beta != 0 else np.nan

def omega_ratio(returns, threshold=0):
    gain = returns[returns > threshold]
    loss = abs(returns[returns < threshold])
    return gain.sum() / loss.sum() if loss.sum() > 0 else np.nan


def optimize_weights(prices):
    n = prices.shape[1]
    def neg_sharpe(w):
        _, ret = backtest_equity_curve(prices, w)
        return -sharpe_ratio(ret)
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * n
    res = scipy.optimize.minimize(neg_sharpe, np.ones(n)/n, bounds=bounds, constraints=cons)
    return res.x if res.success else np.ones(n)/n

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Helper for interactive table
def interactive_table(df):
    st.dataframe(df, use_container_width=True)
    st.markdown("**Sort columns by clicking headers. Use filter box to search.**")

st.set_page_config(page_title="Multi-Factor Quant Dashboard", layout="wide")

# Sidebar tabs for navigation
st.set_page_config(page_title="Multi-Factor Quant Dashboard", layout="wide")
sidebar_section = st.sidebar.radio("Section", ["User Inputs", "Portfolio"])

if sidebar_section == "User Inputs":
    sidebar_tabs = st.sidebar.radio("User Inputs", ["Stock Input/CSV Upload"])
elif sidebar_section == "Portfolio":
    sidebar_tabs = st.sidebar.radio("Portfolio", [
        "Portfolio Overview", "Factor Analysis", "Rolling Beta", "Simulation", "Alpha Generators"
    ])
# ---------------------------
# Alpha Generators Tab
# ---------------------------
if sidebar_tabs == "Alpha Generators":
    st.title("Arcadia")
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    nltk.download('vader_lexicon', quiet=True)
    portfolio = st.session_state.get('portfolio', None)
    if portfolio is None:
        st.info("No portfolio data found. Please input and save your portfolio in the Stock Input/CSV Upload tab.")
    else:
        tabs = st.tabs(["Summary", "News & Sentiment", "Multifactor", "Fetrosky"])

        # --- Modular Functions ---
        def fetch_data(ticker):
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1y")
            news = getattr(stock, 'news', [])
            return info, hist, news

        def compute_factors(info, hist):
            # Value
            value = [info.get('trailingPE', np.nan), info.get('priceToBook', np.nan), info.get('priceToSalesTrailing12Months', np.nan)]
            # Growth
            growth = [info.get('revenueGrowth', np.nan), info.get('earningsGrowth', np.nan), info.get('forwardEps', np.nan)]
            # Revisions
            revisions = [info.get('targetMeanPrice', np.nan), info.get('recommendationMean', np.nan)]
            # Momentum
            try:
                m_52w = hist['Close'].iloc[-1] / info.get('fiftyTwoWeekHigh', np.nan) if info.get('fiftyTwoWeekHigh', np.nan) else np.nan
            except:
                m_52w = np.nan
            momentum = [m_52w,
                        hist['Close'].pct_change(63).iloc[-1] if len(hist) > 63 else np.nan,
                        hist['Close'].pct_change(252).iloc[-1] if len(hist) > 252 else np.nan]
            # Risk
            risk = [info.get('beta', np.nan), info.get('debtToEquity', np.nan), info.get('dividendYield', np.nan)]
            # Quality
            quality = [info.get('returnOnEquity', np.nan), info.get('returnOnAssets', np.nan), info.get('operatingMargins', np.nan), info.get('grossMargins', np.nan)]
            return value, growth, revisions, momentum, risk, quality

        def normalize(arr):
            arr = np.array(arr, dtype=np.float64)
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                return 0
            return (arr - np.mean(arr)) / (np.std(arr) if np.std(arr) > 0 else 1)

        def compute_alpha_strength(value, growth, revisions, momentum, risk, quality):
            scores = [np.nanmean(normalize(value)), np.nanmean(normalize(growth)), np.nanmean(normalize(revisions)),
                      np.nanmean(normalize(momentum)), np.nanmean(normalize(risk)), np.nanmean(normalize(quality))]
            composite = np.nanmean(scores)
            return composite, scores

        def compute_sentiment(news):
            sid = SentimentIntensityAnalyzer()
            sentiments = []
            dates = []
            for item in news:
                headline = item.get('title', '')
                dt = item.get('providerPublishTime', None)
                score = sid.polarity_scores(headline)['compound']
                sentiments.append(score)
                dates.append(datetime.fromtimestamp(dt) if dt else None)
            df = pd.DataFrame({'date': dates, 'sentiment': sentiments}).dropna()
            return df

        def fetrosky_pairs(portfolio):
            tickers = portfolio['ticker'].tolist()
            pairs = []
            for i in range(len(tickers)):
                for j in range(i+1, len(tickers)):
                    t1, t2 = tickers[i], tickers[j]
                    h1 = yf.Ticker(t1).history(period="1y")['Close']
                    h2 = yf.Ticker(t2).history(period="1y")['Close']
                    df = pd.DataFrame({'x': h1, 'y': h2}).dropna()
                    if len(df) < 60:
                        continue
                    x, y = df['x'], df['y']
                    beta = np.cov(x, y)[0,1] / np.var(y) if np.var(y) > 0 else 0
                    spread = x - beta*y
                    # Engle-Granger cointegration test
                    from numpy.linalg import lstsq
                    res = lstsq(y.values.reshape(-1,1), x.values, rcond=None)
                    residuals = x.values - res[0][0]*y.values
                    from statsmodels.tsa.stattools import adfuller
                    pval = adfuller(residuals)[1] if len(residuals) > 0 else 1
                    if pval < 0.05:
                        pairs.append({'pair': (t1, t2), 'spread': spread, 'pval': pval})
            return pairs

        # --- Tab 1: Summary ---
        with tabs[0]:
            st.subheader("Alpha Strength Summary")
            alpha_scores = []
            for idx, row in portfolio.iterrows():
                ticker = row['ticker']
                info, hist, news = fetch_data(ticker)
                value, growth, revisions, momentum, risk, quality = compute_factors(info, hist)
                composite, scores = compute_alpha_strength(value, growth, revisions, momentum, risk, quality)
                alpha_scores.append({'Ticker': ticker, 'Alpha Strength': round(50+50*composite,1)})
            # Heatmap
            df_alpha = pd.DataFrame(alpha_scores)
            z = [df_alpha['Alpha Strength'].values.tolist()]
            x = df_alpha['Ticker'].values.tolist()
            y = ['Alpha Strength']
            fig = ff.create_annotated_heatmap(z=z, x=x, y=y, colorscale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
            # Ranking table
            df_alpha = df_alpha.sort_values('Alpha Strength', ascending=False)
            st.dataframe(df_alpha, use_container_width=True)

        # --- Tab 2: News & Sentiment ---
        with tabs[1]:
            st.subheader("News & Sentiment Analysis")
            sid = SentimentIntensityAnalyzer()
            for idx, row in portfolio.iterrows():
                ticker = row['ticker']
                info, hist, news = fetch_data(ticker)
                st.markdown(f"**{ticker} Headlines:**")
                headlines = [item.get('title','') for item in news]
                st.write(headlines)
                df_sent = compute_sentiment(news)
                if not df_sent.empty:
                    # Sentiment timeline
                    fig = px.line(df_sent, x='date', y='sentiment', title=f"Sentiment Timeline for {ticker}")
                    st.plotly_chart(fig, use_container_width=True)
                    # Sentiment distribution
                    df_sent['label'] = pd.cut(df_sent['sentiment'], bins=[-1, -0.05, 0.05, 1], labels=['Negative','Neutral','Positive'])
                    bar_fig = px.bar(df_sent['label'].value_counts(), title=f"Sentiment Distribution for {ticker}")
                    st.plotly_chart(bar_fig, use_container_width=True)
                    # Alert for sentiment drop
                    roll_mean = df_sent['sentiment'].rolling(10).mean()
                    roll_std = df_sent['sentiment'].rolling(10).std()
                    if len(roll_mean) > 10 and (df_sent['sentiment'].iloc[-1] < roll_mean.iloc[-1] - 2*roll_std.iloc[-1]):
                        st.error(f"Alert: Sentiment for {ticker} dropped more than 2σ below rolling mean!")

        # --- Tab 3: Multifactor Model ---
        with tabs[2]:
            st.subheader("Multifactor Model Breakdown")
            factor_table = []
            radar_data = {}
            for idx, row in portfolio.iterrows():
                ticker = row['ticker']
                info, hist, news = fetch_data(ticker)
                value, growth, revisions, momentum, risk, quality = compute_factors(info, hist)
                composite, scores = compute_alpha_strength(value, growth, revisions, momentum, risk, quality)
                factor_table.append({
                    'Ticker': ticker,
                    'Value': scores[0],
                    'Growth': scores[1],
                    'Revisions': scores[2],
                    'Momentum': scores[3],
                    'Risk': scores[4],
                    'Quality': scores[5],
                    'Composite': composite
                })
                radar_data[ticker] = scores
            df_factors = pd.DataFrame(factor_table).sort_values('Composite', ascending=False)
            st.dataframe(df_factors, use_container_width=True)
            # Radar chart for each stock
            for ticker, scores in radar_data.items():
                fig = go.Figure()
                categories = ['Value','Growth','Revisions','Momentum','Risk','Quality']
                fig.add_trace(go.Scatterpolar(r=scores, theta=categories, fill='toself', name=ticker))
                fig.update_layout(title=f"Factor Radar Chart: {ticker}", polar=dict(radialaxis=dict(visible=True)), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # --- Tab 4: Fetrosky Mean Reversion ---
        with tabs[3]:
            st.subheader("Fetrosky-Style Mean Reversion Pairs")
            pairs = fetrosky_pairs(portfolio)
            if pairs:
                for pair in pairs:
                    t1, t2 = pair['pair']
                    spread = pair['spread']
                    fig = px.line(x=spread.index, y=spread.values, title=f"Spread Plot: {t1} vs {t2}")
                    st.plotly_chart(fig, use_container_width=True)
                    st.success(f"Cointegrated pair: {t1} & {t2} (p-value={pair['pval']:.3f})")
            else:
                st.info("No cointegrated pairs found.")


# ---------------------------
# 1. Upload Portfolio
# ---------------------------
st.title("Arcadia")
# Portfolio input logic
portfolio = st.session_state.get('portfolio', None)
portfolio_data = {}
factor_data = []
portfolio_metrics = None

# New: Add more stocks and dollar cost averaging
if sidebar_tabs == "Stock Input/CSV Upload":
    st.markdown("---")
    st.subheader("Saved Portfolio Data (Session State)")
    saved_portfolio = st.session_state.get('portfolio', None)
    if saved_portfolio is not None:
        st.write(saved_portfolio)
    else:
        st.info("No portfolio data currently saved.")
    st.subheader("Portfolio Input")
    upload_col, manual_col = st.columns(2)
    with upload_col:
        st.markdown("**Upload Portfolio CSV**")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            try:
                portfolio = pd.read_csv(uploaded_file)
                st.session_state['portfolio'] = portfolio.copy()
                st.success("CSV uploaded and saved successfully!")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    with manual_col:
        st.markdown("**Manual Entry**")
        with st.form(key="portfolio_form"):
            tickers = st.text_input("Tickers (comma separated)", value="AAPL,MSFT,GOOGL")
            buy_prices = st.text_input("Buy Prices (comma separated)", value="150,250,120")
            buy_dates = st.text_input("Buy Dates (comma separated, YYYY-MM-DD)", value="2023-01-10,2023-02-15,2023-03-20")
            quantities = st.text_input("Quantities (comma separated)", value="10,5,8")
            submit_portfolio = st.form_submit_button("Submit Portfolio")
        if submit_portfolio:
            try:
                ticker_list = [t.strip() for t in tickers.split(",")]
                price_list = [float(p.strip()) for p in buy_prices.split(",")]
                date_list = [d.strip() for d in buy_dates.split(",")]
                quantity_list = [int(q.strip()) for q in quantities.split(",")]
                portfolio = pd.DataFrame({
                    "ticker": ticker_list,
                    "buy_price": price_list,
                    "buy_date": date_list,
                    "quantity": quantity_list
                })
                st.session_state['portfolio'] = portfolio.copy()
                st.success("Portfolio entered and saved successfully!")
            except Exception as e:
                st.error(f"Error parsing portfolio input: {e}")
    # Add More Stocks (DCA) ONLY in User Inputs
    st.markdown("---")
    st.markdown("### Add More Stocks (Dollar Cost Averaging)")
    with st.form(key="add_stocks_form"):
        new_tickers = st.text_input("New Tickers (comma separated)", value="")
        new_buy_prices = st.text_input("New Buy Prices (comma separated)", value="")
        new_buy_dates = st.text_input("New Buy Dates (comma separated, YYYY-MM-DD)", value="")
        new_quantities = st.text_input("New Quantities (comma separated)", value="")
        submit_new_stocks = st.form_submit_button("Add Stocks")
    if submit_new_stocks:
        try:
            ticker_list = [t.strip() for t in new_tickers.split(",") if t.strip()]
            price_list = [float(p.strip()) for p in new_buy_prices.split(",") if p.strip()]
            date_list = [d.strip() for d in new_buy_dates.split(",") if d.strip()]
            quantity_list = [int(q.strip()) for q in new_quantities.split(",") if q.strip()]
            new_df = pd.DataFrame({
                "ticker": ticker_list,
                "buy_price": price_list,
                "buy_date": date_list,
                "quantity": quantity_list
            })
            # Dollar cost averaging: consolidate with existing portfolio
            if st.session_state.get('portfolio', None) is not None:
                portfolio = st.session_state['portfolio']
                combined = pd.concat([portfolio, new_df], ignore_index=True)
                # Group by ticker and calculate weighted average buy price and total quantity
                consolidated = combined.groupby('ticker').apply(
                    lambda x: pd.Series({
                        'buy_price': np.average(x['buy_price'], weights=x['quantity']),
                        'buy_date': x['buy_date'].iloc[-1], # most recent buy date
                        'quantity': x['quantity'].sum()
                    })
                ).reset_index()
                st.session_state['portfolio'] = consolidated
                st.success("Stocks added and portfolio consolidated via dollar cost averaging!")
            else:
                st.session_state['portfolio'] = new_df.copy()
                st.success("Stocks added as new portfolio!")
        except Exception as e:
            st.error(f"Error adding stocks: {e}")
    portfolio = st.session_state.get('portfolio', None)
    if portfolio is not None:
        st.subheader("Portfolio Table")
        interactive_table(portfolio)
        delete_col = st.columns([1])[0]
        with delete_col:
            if st.button("Delete Portfolio Data"):
                st.session_state.pop('portfolio', None)
                st.success("Portfolio data deleted.")

# Portfolio Overview Tab: Fetch OHLCV and compute metrics
if sidebar_tabs == "Portfolio Overview":
    portfolio = st.session_state.get('portfolio', None)
    if portfolio is None:
        st.info("No portfolio data found. Please input and save your portfolio in the Stock Input/CSV Upload tab.")
    else:
        st.subheader("Portfolio Metrics & Performance")
        portfolio_data = {}
        metrics = []
        total_value = 0
        weighted_return = 0
        for idx, row in portfolio.iterrows():
            ticker = row['ticker']
            buy_price = row['buy_price']
            buy_date = row['buy_date']
            quantity = row['quantity']
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=buy_date)
                info = stock.info
                portfolio_data[ticker] = hist
                current_price = hist['Close'].iloc[-1] if len(hist) > 0 else np.nan
                value = current_price * quantity if not np.isnan(current_price) else np.nan
                pl = (current_price - buy_price) * quantity if not np.isnan(current_price) else np.nan
                ret = (current_price - buy_price)/buy_price if not np.isnan(current_price) else np.nan
                weighted_return += ret * value if not np.isnan(ret) and not np.isnan(value) else 0
                total_value += value if not np.isnan(value) else 0
                # Volatility (annualized std of daily returns)
                volatility = hist['Close'].pct_change().std() * np.sqrt(252) if len(hist) > 1 else np.nan
                # Max drawdown
                roll_max = hist['Close'].cummax() if len(hist) > 1 else pd.Series([np.nan])
                drawdown = (hist['Close'] - roll_max) / roll_max if len(hist) > 1 else pd.Series([np.nan])
                max_dd_local = drawdown.min() if len(drawdown) > 0 else np.nan
                metrics.append({
                    'Ticker': ticker,
                    'Current Price': current_price,
                    'Value': value,
                    'Unrealized P/L': pl,
                    'Return %': ret * 100 if not np.isnan(ret) else np.nan,
                    'Volatility': volatility,
                    'Max Drawdown': max_dd_local
                })
            except Exception as e:
                metrics.append({
                    'Ticker': ticker,
                    'Current Price': np.nan,
                    'Value': np.nan,
                    'Unrealized P/L': np.nan,
                    'Return %': np.nan,
                    'Volatility': np.nan,
                    'Max Drawdown': np.nan
                })
        metrics_df = pd.DataFrame(metrics)
        # Weighted return
        weighted_return = weighted_return / total_value if total_value > 0 else np.nan
        # Portfolio ratios (using equally weighted portfolio)
        tickers = portfolio['ticker'].tolist()
        price_data = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="2y")
                price_data[ticker] = hist['Close']
            except Exception:
                continue
        if price_data:
            prices_df = pd.DataFrame(price_data).dropna()
            weights = np.ones(len(tickers))/len(tickers)
            equity_curve, portfolio_ret = backtest_equity_curve(prices_df, weights)
            # Use S&P 500 as benchmark if available
            try:
                sp500 = yf.Ticker("^GSPC").history(period="2y")['Close']
                sp500 = sp500.loc[prices_df.index]
                info_ratio = information_ratio(portfolio_ret, sp500.pct_change().fillna(0))
            except Exception:
                info_ratio = np.nan
            sortino = sortino_ratio(portfolio_ret)
            calmar = calmar_ratio(portfolio_ret, equity_curve)
            omega = omega_ratio(portfolio_ret)
            # Estimate beta for Treynor (regression vs S&P 500)
            try:
                cov = np.cov(portfolio_ret, sp500.pct_change().fillna(0))[0,1]
                var = np.var(sp500.pct_change().fillna(0))
                beta = cov / var if var > 0 else np.nan
                treynor = treynor_ratio(portfolio_ret, beta)
            except Exception:
                beta = np.nan
                treynor = np.nan
        else:
            info_ratio = sortino = calmar = omega = beta = treynor = np.nan
        # Dashboard cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Portfolio Value", f"${total_value:,.2f}")
        col2.metric("Weighted Return %", f"{weighted_return*100:.2f}%", delta=None)
        col3.metric("Best Stock Return %", f"{metrics_df['Return %'].max():.2f}%", delta=None)
        col4.metric("Worst Stock Return %", f"{metrics_df['Return %'].min():.2f}%", delta=None)
        st.markdown("### Portfolio Ratios")
        st.markdown(f"- **Sharpe Ratio:** `{sharpe_ratio(portfolio_ret):.3f}`")
        st.markdown(f"- **Sortino Ratio:** `{sortino:.3f}`")
        st.markdown(f"- **Information Ratio:** `{info_ratio:.3f}`")
        st.markdown(f"- **Calmar Ratio:** `{calmar:.3f}`")
        st.markdown(f"- **Treynor Ratio:** `{treynor:.3f}` (Beta: {beta:.3f})")
        st.markdown(f"- **Omega Ratio:** `{omega:.3f}`")
        st.markdown("### Stock Metrics Table")
        def color_pnl(val):
            if np.isnan(val):
                return ''
            color = 'green' if val > 0 else 'red'
            return f'background-color: {color}; color: white;'
        styled_df = metrics_df.style.applymap(color_pnl, subset=['Unrealized P/L'])
        st.dataframe(styled_df, use_container_width=True)

# Rolling Beta Tab
# Simulation Tab: Brownian Motion Monte Carlo
if sidebar_tabs == "Simulation":
    portfolio = st.session_state.get('portfolio', None)
    if portfolio is None:
        st.info("No portfolio data found. Please input and save your portfolio in the Stock Input/CSV Upload tab.")
    else:
        st.subheader("Geometric Brownian Motion Monte Carlo Simulation")
        num_sim = st.slider("Number of Simulations", min_value=100, max_value=2000, value=1000, step=100)
        target_return = st.number_input("Target Return (%)", min_value=-100.0, max_value=500.0, value=10.0, step=1.0)
        sim_days = st.slider("Simulation Horizon (days)", min_value=30, max_value=365, value=252, step=1)
        for idx, row in portfolio.iterrows():
            ticker = row['ticker']
            buy_price = row['buy_price']
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="2y")
                S0 = hist['Close'].iloc[-1]
                mu = hist['Close'].pct_change().mean() * 252  # annualized drift
                sigma = hist['Close'].pct_change().std() * np.sqrt(252)  # annualized volatility
                dt = 1/252
                N = sim_days
                M = num_sim
                price_paths = np.zeros((N, M))
                for m in range(M):
                    prices = [S0]
                    for i in range(1, N):
                        dS = prices[-1] * (mu*dt + sigma*np.random.normal()*np.sqrt(dt))
                        prices.append(prices[-1]+dS)
                    price_paths[:, m] = prices
                # Plot simulation paths
                import plotly.graph_objects as go
                fig = go.Figure()
                for m in range(min(M, 50)):
                    fig.add_trace(go.Scatter(y=price_paths[:, m], mode='lines', line=dict(width=1), name=f"Sim {m+1}"))
                fig.update_layout(title=f"Simulated Price Paths for {ticker}", xaxis_title="Days", yaxis_title="Price")
                st.plotly_chart(fig, use_container_width=True)
                # Probability distribution of final returns
                final_prices = price_paths[-1, :]
                final_returns = (final_prices - S0) / S0 * 100
                import plotly.express as px
                hist_fig = px.histogram(final_returns, nbins=50, title=f"Distribution of Final Returns for {ticker}")
                st.plotly_chart(hist_fig, use_container_width=True)
                prob_beating_target = np.mean(final_returns > target_return)
                st.markdown(f"**Probability of Beating Target Return ({target_return}%): {prob_beating_target*100:.1f}%**")
            except Exception as e:
                st.warning(f"Simulation failed for {ticker}: {e}")
import scipy.optimize
def backtest_equity_curve(prices, weights):
    returns = prices.pct_change().fillna(0)
    portfolio_ret = (returns * weights).sum(axis=1)
    equity_curve = (1 + portfolio_ret).cumprod()
    return equity_curve, portfolio_ret
def sharpe_ratio(returns):
    return returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan

def optimize_weights(prices):
    n = prices.shape[1]
    def neg_sharpe(w):
        _, ret = backtest_equity_curve(prices, w)
        return -sharpe_ratio(ret)
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * n
    res = scipy.optimize.minimize(neg_sharpe, np.ones(n)/n, bounds=bounds, constraints=cons)
    return res.x if res.success else np.ones(n)/n
if sidebar_tabs == "Factor Analysis":
    portfolio = st.session_state.get('portfolio', None)
    if portfolio is None:
        st.info("No portfolio data found. Please input and save your portfolio in the Stock Input/CSV Upload tab.")
    else:
        factor_categories = {
            'Value': ['pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_ebitda', 'ev_sales', 'price_cash_flow'],
            'Growth': ['revenue_growth', 'eps_growth', 'revenue_growth_5y', 'eps_growth_5y', 'revenue_accel', 'eps_accel', 'fcf_growth', 'ebitda_growth', 'net_income_growth'],
            'Revisions': ['target_price_change', 'analyst_rec', 'num_upgrades', 'eps_revisions', 'num_buy_hold_sell', 'change_mean_target', 'estimate_revisions'],
            'Momentum': ['momentum_1m', 'momentum_3m', 'momentum_6m', 'momentum_12m', 'fifty_two_week_high', 'vwap', 'rsi', 'macd_signal'],
            'Risk': ['beta', 'volatility', 'net_debt_equity', 'dividend_yield', 'current_ratio', 'quick_ratio', 'max_drawdown', 'altman_z'],
            'Quality': ['roe', 'roa', 'operating_margin', 'gross_profit_assets', 'market_cap', 'fcf_assets', 'profit_margin', 'ebit_margin']
        }
        # Set factor weights UI (sliders)
        st.markdown("**Set Factor Weights (Total = 100%)**")
        factor_weights = {}
        total_weight = 0
        cols = st.columns(6)
        for i, cat in enumerate(factor_categories.keys()):
            factor_weights[cat] = cols[i].slider(f"{cat} Weight (%)", min_value=0, max_value=100, value=round(100/6), step=1)
            total_weight += factor_weights[cat]
        # Normalize weights to sum to 1
        if total_weight == 0:
            normalized_weights = {cat: 1.0/6 for cat in factor_categories.keys()}
        else:
            normalized_weights = {cat: factor_weights[cat]/total_weight for cat in factor_categories.keys()}
        st.markdown(f"**Total Weight:** {total_weight}%")
        if total_weight != 100:
            st.warning("Total weight should be 100%. Adjust sliders.")
        # Suggest optimal factor weights if portfolio optimization was run
        optimal_weights = st.session_state.get('optimal_weights', None)
        if optimal_weights is not None and len(optimal_weights) == len(portfolio):
            st.markdown("---")
            st.markdown("#### Suggested Factor Weights (based on portfolio optimization)")
            # Calculate which factors contributed most to top-ranked stocks by optimal weight
            # Get top stocks by optimal weight
            top_idx = np.argsort(optimal_weights)[::-1][:max(1, len(optimal_weights)//2)]
            top_tickers = portfolio.iloc[top_idx]['ticker'].tolist()
            # Aggregate factor scores for top stocks
            factor_data = []
            price_data = {}
            for idx, row in portfolio.iterrows():
                ticker = row['ticker']
                buy_date = row['buy_date']
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(start=buy_date)
                    info = stock.info
                    price_data[ticker] = hist['Close']
                    factor_row = {
                        'ticker': ticker,
                        # Value
                        'pe_ratio': info.get('trailingPE', np.nan),
                        'pb_ratio': info.get('priceToBook', np.nan),
                        'ps_ratio': info.get('priceToSalesTrailing12Months', np.nan),
                        'ev_ebitda': info.get('enterpriseToEbitda', np.nan),
                        'ev_sales': info.get('enterpriseToRevenue', np.nan),
                        'price_cash_flow': info.get('priceToCashflow', np.nan),
                        # Growth
                        'revenue_growth': info.get('revenueGrowth', np.nan),
                        'eps_growth': info.get('earningsGrowth', np.nan),
                        # Revisions
                        'target_price_change': info.get('targetMeanPrice', np.nan),
                        'analyst_rec': info.get('recommendationMean', np.nan),
                        # Momentum
                        'momentum_1m': hist['Close'].pct_change(21).iloc[-1] if len(hist) > 21 else np.nan,
                        'momentum_3m': hist['Close'].pct_change(63).iloc[-1] if len(hist) > 63 else np.nan,
                        'momentum_6m': hist['Close'].pct_change(126).iloc[-1] if len(hist) > 126 else np.nan,
                        'momentum_12m': hist['Close'].pct_change(252).iloc[-1] if len(hist) > 252 else np.nan,
                        'fifty_two_week_high': info.get('fiftyTwoWeekHigh', np.nan),
                        # Risk
                        'beta': info.get('beta', np.nan),
                        'volatility': hist['Close'].pct_change().std() * np.sqrt(252) if len(hist) > 1 else np.nan,
                        'max_drawdown': ((hist['Close'] - hist['Close'].cummax()) / hist['Close'].cummax()).min() if len(hist) > 1 else np.nan,
                        # Quality
                        'roe': info.get('returnOnEquity', np.nan),
                        'roa': info.get('returnOnAssets', np.nan),
                        'operating_margin': info.get('operatingMargins', np.nan),
                        'market_cap': info.get('marketCap', np.nan)
                    }
                    factor_data.append(factor_row)
                except Exception as e:
                    st.warning(f"Failed to fetch factor data for {ticker}: {e}")
            data = pd.DataFrame(factor_data)
            # Normalize and score by category
            category_scores = {}
            for cat, factors in factor_categories.items():
                available = [f for f in factors if f in data.columns]
                if available:
                    norm = (data[available] - data[available].mean()) / data[available].std()
                    category_scores[cat] = norm.mean(axis=1)
                else:
                    category_scores[cat] = pd.Series([np.nan]*len(data))
            # Aggregate factor scores for top stocks
            suggested_weights = {}
            for cat in factor_categories.keys():
                # Average score for top stocks in this category
                scores = category_scores[cat][top_idx]
                suggested_weights[cat] = np.nanmean(scores) if len(scores) > 0 else 0
            # Normalize suggested weights to sum to 100%
            total_suggested = sum([v for v in suggested_weights.values() if not np.isnan(v)])
            if total_suggested > 0:
                for cat in suggested_weights:
                    suggested_weights[cat] = round(100 * suggested_weights[cat] / total_suggested, 1)
            else:
                for cat in suggested_weights:
                    suggested_weights[cat] = round(100/6, 1)
            st.write(suggested_weights)
            st.info("These suggested weights are based on the average factor scores of your top-optimized stocks. You can use them to adjust your sliders above.")
        # ...existing code for factor data, scoring, and Stock Factor Rankings...
        factor_data = []
        price_data = {}
        for idx, row in portfolio.iterrows():
            ticker = row['ticker']
            buy_date = row['buy_date']
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=buy_date)
                info = stock.info
                price_data[ticker] = hist['Close']
                factor_row = {
                    'ticker': ticker,
                    # Value
                    'pe_ratio': info.get('trailingPE', np.nan),
                    'pb_ratio': info.get('priceToBook', np.nan),
                    'ps_ratio': info.get('priceToSalesTrailing12Months', np.nan),
                    'ev_ebitda': info.get('enterpriseToEbitda', np.nan),
                    'ev_sales': info.get('enterpriseToRevenue', np.nan),
                    'price_cash_flow': info.get('priceToCashflow', np.nan),
                    # Growth
                    'revenue_growth': info.get('revenueGrowth', np.nan),
                    'eps_growth': info.get('earningsGrowth', np.nan),
                    # Revisions
                    'target_price_change': info.get('targetMeanPrice', np.nan),
                    'analyst_rec': info.get('recommendationMean', np.nan),
                    # Momentum
                    'momentum_1m': hist['Close'].pct_change(21).iloc[-1] if len(hist) > 21 else np.nan,
                    'momentum_3m': hist['Close'].pct_change(63).iloc[-1] if len(hist) > 63 else np.nan,
                    'momentum_6m': hist['Close'].pct_change(126).iloc[-1] if len(hist) > 126 else np.nan,
                    'momentum_12m': hist['Close'].pct_change(252).iloc[-1] if len(hist) > 252 else np.nan,
                    'fifty_two_week_high': info.get('fiftyTwoWeekHigh', np.nan),
                    # Risk
                    'beta': info.get('beta', np.nan),
                    'volatility': hist['Close'].pct_change().std() * np.sqrt(252) if len(hist) > 1 else np.nan,
                    'max_drawdown': ((hist['Close'] - hist['Close'].cummax()) / hist['Close'].cummax()).min() if len(hist) > 1 else np.nan,
                    # Quality
                    'roe': info.get('returnOnEquity', np.nan),
                    'roa': info.get('returnOnAssets', np.nan),
                    'operating_margin': info.get('operatingMargins', np.nan),
                    'market_cap': info.get('marketCap', np.nan)
                }
                factor_data.append(factor_row)
            except Exception as e:
                st.warning(f"Failed to fetch factor data for {ticker}: {e}")
        data = pd.DataFrame(factor_data)
        if not data.empty:
            # Normalize and score by category
            category_scores = {}
            subfactor_scores = {}
            for cat, factors in factor_categories.items():
                available = [f for f in factors if f in data.columns]
                if available:
                    norm = (data[available] - data[available].mean()) / data[available].std()
                    category_scores[cat] = norm.mean(axis=1)
                    # Save subfactor scores for UX
                    for f in available:
                        subfactor_scores[f] = norm[f]
                else:
                    category_scores[cat] = pd.Series([np.nan]*len(data))
                    for f in factors:
                        subfactor_scores[f] = pd.Series([np.nan]*len(data))
            # Composite score (use user weights)
            composite = np.zeros(len(data))
            for cat in factor_categories.keys():
                composite += category_scores[cat] * normalized_weights[cat]
            data['Composite Score'] = composite
            # Add category scores to table
            for cat in category_scores:
                data[f'{cat} Score'] = category_scores[cat]
            # Add subfactor scores to table for UX feedback
            for f in subfactor_scores:
                data[f] = subfactor_scores[f]
            # Rank and highlight
            data = data.sort_values('Composite Score', ascending=False)
            st.markdown("### Stock Factor Rankings")
            show_cols = ['ticker', 'Composite Score'] + [f'{cat} Score' for cat in factor_categories.keys()]
            styled = data[show_cols].style.applymap(
                lambda val: 'background-color: green; color: white;' if val == data['Composite Score'].max() else ('background-color: red; color: white;' if val == data['Composite Score'].min() else ''),
                subset=['Composite Score']
            )
            st.dataframe(styled, use_container_width=True)
            # Show subfactor scores below for UX feedback
            st.markdown("#### Subfactor Scores (see impact of sliders)")
            subfactor_cols = ['ticker'] + list(subfactor_scores.keys())
            st.dataframe(data[subfactor_cols], use_container_width=True)
            # Backtest and optimized weights UI
            st.markdown("---")
            st.subheader("Backtest and Optimize Portfolio Weights")
            st.markdown("**Objective:** Maximize Sharpe Ratio (mean return / volatility)")
            # Prepare prices_df for optimization
            if price_data:
                prices_df = pd.DataFrame(price_data).dropna()
            else:
                prices_df = None
            if st.button("Run Optimization"):
                with st.spinner("Running optimization..."):
                    try:
                        if prices_df is not None and not prices_df.empty:
                            optimal_weights = optimize_weights(prices_df)
                            st.session_state['optimal_weights'] = optimal_weights
                            st.success("Optimization complete!")
                        else:
                            st.error("Not enough price data for optimization.")
                    except Exception as e:
                        st.error(f"Error running optimization: {e}")
            optimal_weights = st.session_state.get('optimal_weights', None)
            if optimal_weights is not None and len(optimal_weights) == len(portfolio):
                st.markdown("### Optimal Weights (Max Sharpe Ratio)")
                weights_df = pd.DataFrame({
                    'Ticker': portfolio['ticker'],
                    'Optimal Weight': optimal_weights
                })
                st.dataframe(weights_df)
                # Plot optimal vs. equal weights
                st.markdown("### Portfolio Performance: Optimal Weights vs. Equal Weights")
                weights_equal = np.ones(len(portfolio)) / len(portfolio)
                _, ret_opt = backtest_equity_curve(prices_df, optimal_weights)
                _, ret_equal = backtest_equity_curve(prices_df, weights_equal)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ret_opt.index, y=ret_opt, mode='lines', name='Optimal Weights'))
                fig.add_trace(go.Scatter(x=ret_equal.index, y=ret_equal, mode='lines', name='Equal Weights'))
                fig.update_layout(title="Cumulative Return: Optimal Weights vs. Equal Weights", xaxis_title="Date", yaxis_title="Cumulative Return")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run the optimization to see optimal weights and performance comparison.")
        else:
            st.warning("No factor data available for analysis.")
# ---------------------------
# 3. Example: Pull fundamental data
# ---------------------------
if sidebar_tabs == "Factor Analysis":
    portfolio = st.session_state.get('portfolio', None)
    if portfolio is None:
        st.info("No portfolio data found. Please input and save your portfolio in the Stock Input/CSV Upload tab.")
    else:
        st.subheader("Multi-Factor Model Analysis")
        st.markdown("**Set Factor Weights (Multi-Factor Categories)**")
        st.markdown("Adjust the importance of each factor category (0 = ignore, 1 = max weight). All weights will be normalized automatically.")
        factor_categories = {
            'Value': ['pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_ebitda', 'ev_sales', 'price_cash_flow'],
            'Growth': ['revenue_growth', 'eps_growth', 'revenue_growth_5y', 'eps_growth_5y', 'revenue_accel', 'eps_accel', 'fcf_growth', 'ebitda_growth', 'net_income_growth'],
            'Revisions': ['target_price_change', 'analyst_rec', 'num_upgrades', 'eps_revisions', 'num_buy_hold_sell', 'change_mean_target', 'estimate_revisions'],
            'Momentum': ['momentum_1m', 'momentum_3m', 'momentum_6m', 'momentum_12m', 'fifty_two_week_high', 'vwap', 'rsi', 'macd_signal'],
            'Risk': ['beta', 'volatility', 'net_debt_equity', 'dividend_yield', 'current_ratio', 'quick_ratio', 'max_drawdown', 'altman_z'],
            'Quality': ['roe', 'roa', 'operating_margin', 'gross_profit_assets', 'market_cap', 'fcf_assets', 'profit_margin', 'ebit_margin']
        }
        factor_data = []
        for idx, row in portfolio.iterrows():
            ticker = row['ticker']
            buy_date = row['buy_date']
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=buy_date)
                info = stock.info
                factor_row = {
                    'ticker': ticker,
                    # Value
                    'pe_ratio': info.get('trailingPE', np.nan),
                    'pb_ratio': info.get('priceToBook', np.nan),
                    'ps_ratio': info.get('priceToSalesTrailing12Months', np.nan),
                    'ev_ebitda': info.get('enterpriseToEbitda', np.nan),
                    'ev_sales': info.get('enterpriseToRevenue', np.nan),
                    'price_cash_flow': info.get('priceToCashflow', np.nan),
                    # Growth
                    'revenue_growth': info.get('revenueGrowth', np.nan),
                    'eps_growth': info.get('earningsGrowth', np.nan),
                    # Revisions
                    'target_price_change': info.get('targetMeanPrice', np.nan),
                    'analyst_rec': info.get('recommendationMean', np.nan),
                    # Momentum
                    'momentum_1m': hist['Close'].pct_change(21).iloc[-1] if len(hist) > 21 else np.nan,
                    'momentum_3m': hist['Close'].pct_change(63).iloc[-1] if len(hist) > 63 else np.nan,
                    'momentum_6m': hist['Close'].pct_change(126).iloc[-1] if len(hist) > 126 else np.nan,
                    'momentum_12m': hist['Close'].pct_change(252).iloc[-1] if len(hist) > 252 else np.nan,
                    'fifty_two_week_high': info.get('fiftyTwoWeekHigh', np.nan),
                    # Risk
                    'beta': info.get('beta', np.nan),
                    'volatility': hist['Close'].pct_change().std() * np.sqrt(252) if len(hist) > 1 else np.nan,
                    'max_drawdown': ((hist['Close'] - hist['Close'].cummax()) / hist['Close'].cummax()).min() if len(hist) > 1 else np.nan,
                    # Quality
                    'roe': info.get('returnOnEquity', np.nan),
                    'roa': info.get('returnOnAssets', np.nan),
                    'operating_margin': info.get('operatingMargins', np.nan),
                    'market_cap': info.get('marketCap', np.nan)
                }
                factor_data.append(factor_row)
            except Exception as e:
                st.warning(f"Failed to fetch factor data for {ticker}: {e}")
        data = pd.DataFrame(factor_data)
        if not data.empty:
            # Normalize and score by category
            category_scores = {}
            for cat, factors in factor_categories.items():
                available = [f for f in factors if f in data.columns]
                if available:
                    norm = (data[available] - data[available].mean()) / data[available].std()
                    category_scores[cat] = norm.mean(axis=1)
                else:
                    category_scores[cat] = pd.Series([np.nan]*len(data))
            # Composite score (use equal weights since sliders are removed)
            composite = np.zeros(len(data))
            for cat in factor_categories.keys():
                composite += category_scores[cat] * (1.0/len(factor_categories))
            data['Composite Score'] = composite
            # Add category scores to table
            for cat in category_scores:
                data[f'{cat} Score'] = category_scores[cat]
            # Rank and highlight
            data = data.sort_values('Composite Score', ascending=False)
            st.markdown("### Stock Factor Rankings")
            def highlight_top(val):
                if np.isnan(val):
                    return ''
                color = 'green' if val == data['Composite Score'].max() else ('red' if val == data['Composite Score'].min() else '')
                return f'background-color: {color}; color: white;' if color else ''
            styled = data.style.applymap(highlight_top, subset=['Composite Score'])
            st.dataframe(styled, use_container_width=True)
            st.markdown("---")
            # Remove company filter and breakdown UI
            # ...existing code for backtesting and optimized weights...
        else:
            st.warning("No factor data available for analysis.")

# ---------------------------
# 4. Portfolio Metrics (Demo)
# ---------------------------
if portfolio is not None and portfolio_data:
    st.subheader("Portfolio Performance")
    metrics = []
    for idx,row in portfolio.iterrows():
        ticker = row['ticker']
        buy_price = row['buy_price']
        quantity = row['quantity']
        try:
            current_price = portfolio_data[ticker]['Close'].iloc[-1]
            pl = (current_price - buy_price) * quantity
            ret = (current_price - buy_price)/buy_price
            metrics.append([ticker, current_price, pl, ret])
        except Exception as e:
            metrics.append([ticker, np.nan, np.nan, np.nan])
    metrics_df = pd.DataFrame(metrics, columns=['Ticker','Current Price','Unrealized P/L','Return %'])
    st.dataframe(metrics_df)

# ---------------------------
# 5. Brownian Motion Simulation
# ---------------------------
if portfolio is not None and portfolio_data:
    st.subheader("Brownian Motion Price Simulation")
    # Collect summary stats for all stocks first
    summary_rows = []
    sim_results = {}
    for idx, row in portfolio.iterrows():
        ticker = row['ticker']
        buy_price = row['buy_price']
        try:
            hist = portfolio_data[ticker]['Close']
            S0 = hist.iloc[-1]
            mu = hist.pct_change().mean() * 252  # annualized drift
            sigma = hist.pct_change().std() * np.sqrt(252)  # annualized volatility
            T = 1  # 1 year
            dt = 1/252
            N = 252
            M = 50  # number of simulations
            t = np.linspace(0,T,N)
            price_paths = np.zeros((N,M))
            for m in range(M):
                prices = [S0]
                for i in range(1,N):
                    dS = prices[-1] * (mu*dt + sigma*np.random.normal()*np.sqrt(dt))
                    prices.append(prices[-1]+dS)
                price_paths[:,m] = prices
            final_prices = price_paths[-1, :]
            mean_final = np.mean(final_prices)
            median_final = np.median(final_prices)
            min_final = np.min(final_prices)
            max_final = np.max(final_prices)
            prob_gain = np.mean(final_prices > buy_price)
            summary_rows.append({
                'Ticker': ticker,
                'Mean': mean_final,
                'Median': median_final,
                'Min': min_final,
                'Max': max_final,
                'Prob > Buy': prob_gain*100
            })
            sim_results[ticker] = price_paths
        except Exception as e:
            summary_rows.append({
                'Ticker': ticker,
                'Mean': np.nan,
                'Median': np.nan,
                'Min': np.nan,
                'Max': np.nan,
                'Prob > Buy': np.nan
            })
    # Show summary table at top
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        st.markdown("### Simulation Summary Table")
        st.dataframe(summary_df, use_container_width=True)
    # Show individual simulation charts and stats below
    for idx, row in portfolio.iterrows():
        ticker = row['ticker']
        buy_price = row['buy_price']
        if ticker in sim_results:
            price_paths = sim_results[ticker]
            fig = go.Figure()
            for m in range(price_paths.shape[1]):
                fig.add_trace(go.Scatter(y=price_paths[:,m], mode='lines', line=dict(width=1), name=f"Sim {m+1}"))
            fig.update_layout(title=f"Simulated Price Paths for {ticker}", xaxis_title="Days", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)
            final_prices = price_paths[-1, :]
            mean_final = np.mean(final_prices)
            median_final = np.median(final_prices)
            min_final = np.min(final_prices)
            max_final = np.max(final_prices)
            prob_gain = np.mean(final_prices > buy_price)
            st.markdown(f"**{ticker} Simulation Summary:**")
            st.markdown(f"- Mean final price: ${mean_final:,.2f}")
            st.markdown(f"- Median final price: ${median_final:,.2f}")
            st.markdown(f"- Min final price: ${min_final:,.2f}")
            st.markdown(f"- Max final price: ${max_final:,.2f}")
            st.markdown(f"- Probability of finishing above your buy price (${buy_price}): {prob_gain*100:.1f}%")
        else:
            st.warning(f"Simulation failed for {ticker}.")
# ---------------------------
# Rolling Beta Tab: Portfolio Overview + Engle Factor Model
# ---------------------------
if sidebar_tabs == "Rolling Beta":
    portfolio = st.session_state.get('portfolio', None)
    if portfolio is None:
        st.info("No portfolio data found. Please input and save your portfolio in the Stock Input/CSV Upload tab.")
    else:
        st.title("Step 2: Portfolio Overview + Engle-style Rolling Beta Model")
        st.markdown("""
        **Rolling Beta Exposure Model:**
        - Select a stock and benchmark/factor ETF/index
        - Rolling regression (beta, alpha) using yfinance, pandas, numpy
        """)
        # UI: Dropdowns for stock and factor, slider for window size
        tickers = portfolio['ticker'].tolist()
        stock_choice = st.selectbox("Select Stock", tickers)
        factor_options = {
            "SPY (Market)": "SPY",
            "QQQ (Growth)": "QQQ",
            "IWD (Value)": "IWD",
            "EFA (International)": "EFA",
            "XLF (Financials)": "XLF"
        }
        factor_name = st.selectbox("Select Benchmark/Factor", list(factor_options.keys()))
        factor_ticker = factor_options[factor_name]
        window = st.slider("Rolling Window (days)", min_value=30, max_value=252, value=90, step=10)
        # Download price data
        try:
            stock_hist = yf.Ticker(stock_choice).history(period="3y")['Close']
            factor_hist = yf.Ticker(factor_ticker).history(period="3y")['Close']
            # Align dates
            df = pd.DataFrame({
                'stock': stock_hist,
                'factor': factor_hist
            }).dropna()
            # Compute daily log returns
            df['stock_ret'] = np.log(df['stock']).diff()
            df['factor_ret'] = np.log(df['factor']).diff()
            df = df.dropna()
            # Rolling beta and alpha
            rolling_beta = df['stock_ret'].rolling(window).cov(df['factor_ret']) / df['factor_ret'].rolling(window).var()
            rolling_alpha = df['stock_ret'] - rolling_beta * df['factor_ret']
            rolling_alpha = rolling_alpha.rolling(window).mean()
            # Output table
            current_beta = rolling_beta.iloc[-1] if not rolling_beta.empty else np.nan
            avg_beta = rolling_beta.mean() if not rolling_beta.empty else np.nan
            current_alpha = rolling_alpha.iloc[-1] if not rolling_alpha.empty else np.nan
            avg_alpha = rolling_alpha.mean() if not rolling_alpha.empty else np.nan
            st.markdown("#### Rolling Beta Table")
            st.table(pd.DataFrame({
                'Current Beta': [current_beta],
                'Avg Beta': [avg_beta],
                'Current Alpha': [current_alpha],
                'Avg Alpha': [avg_alpha]
            }))
            # Plot chart (matplotlib)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(rolling_beta.index, rolling_beta, label='Rolling Beta')
            ax.set_title(f'Rolling Beta: {stock_choice} vs {factor_ticker}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Beta')
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Rolling beta analysis failed: {e}")