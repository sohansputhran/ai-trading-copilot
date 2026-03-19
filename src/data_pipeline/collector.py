"""
Market Data Collector

WHY THIS MODULE EXISTS:
- We need stock prices to analyze (OHLCV: Open, High, Low, Close, Volume)
- Yahoo Finance is free and doesn't need an API key
- yfinance library makes it easy to fetch data

WHAT IT DOES:
- Fetches historical price data for Indian stocks
- Returns it as a pandas DataFrame (like an Excel table)

HOW TO USE:
    collector = MarketDataCollector()
    data = collector.fetch_data("RELIANCE.NS", period="1mo")
    print(data.head())  # Show first 5 rows
"""


import pandas as pd
import yfinance as yf


class MarketDataCollector:
    """
    Fetches market data from Yahoo Finance.

    For Indian stocks, add .NS suffix (NSE) or .BO (BSE)
    Example: "RELIANCE.NS" for Reliance on NSE
    """

    def fetch_data(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical data for a stock.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE.NS")
            period: How much history? "1d", "5d", "1mo", "3mo", "1y", "max"
            interval: Data frequency? "1m", "5m", "1h", "1d"

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume

        Example:
            collector = MarketDataCollector()
            data = collector.fetch_data("TCS.NS", period="1mo")
        """

        print(f"Fetching data for {symbol}...")

        # Create ticker object
        ticker = yf.Ticker(symbol)

        # Fetch historical data
        df = ticker.history(period=period, interval=interval)

        # Check if we got data
        if df.empty:
            raise ValueError(f"No data found for {symbol}. Check symbol name.")

        print(f"Fetched {len(df)} rows for {symbol}")

        return df

    def fetch_current_price(self, symbol: str) -> float:
        """
        Get just the latest closing price.

        Args:
            symbol: Stock symbol

        Returns:
            Latest close price as float
        """
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1d")

        if df.empty:
            raise ValueError(f"No data for {symbol}")

        return df['Close'].iloc[-1]


# Test the collector
if __name__ == "__main__":
    """
    Test script - run with: python -m src.data_pipeline.collector
    """

    print("\n" + "="*60)
    print("Testing Market Data Collector")
    print("="*60 + "\n")

    collector = MarketDataCollector()

    # Test 1: Fetch 1 month of daily data for Reliance
    print("Test 1: Fetch Reliance (NSE) - 1 month")
    data = collector.fetch_data("RELIANCE.NS", period="1mo")
    print(data.head())
    print(f"\nData shape: {data.shape[0]} days, {data.shape[1]} columns")

    # Test 2: Get current price
    print("\n" + "-"*60)
    print("Test 2: Current price")
    price = collector.fetch_current_price("RELIANCE.NS")
    print(f"RELIANCE current price: ₹{price:.2f}")

    print("\n" + "="*60)
    print("Market Data Collector working!")
    print("="*60)
