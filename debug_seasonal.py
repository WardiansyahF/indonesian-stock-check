import yfinance as yf
import pandas as pd

def test_seasonal(ticker):
    try:
        df = yf.download(ticker, period="5y", interval="1d", progress=False)
        print(f"Index: {type(df.index)}")
        print(f"Columns: {df.columns}")
        
        # Determine how to access Close
        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"][ticker]
        else:
            close = df["Close"]
            
        monthly = close.resample("M").last()
        returns = monthly.pct_change().dropna()
        print(f"Returns head: \n{returns.head()}")
        
        returns_df = returns.to_frame()
        returns_df["Month"] = returns_df.index.month
        avg_returns = returns_df.groupby("Month").mean() * 100
        print(f"Avg returns: \n{avg_returns}")
        return avg_returns
    except Exception as e:
        print(f"Error: {e}")

test_seasonal("BBCA.JK")
