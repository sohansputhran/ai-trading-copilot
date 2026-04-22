def get_current_price(symbol: str) -> float:
    """Fetch current price for a symbol"""
    import yfinance as yf

    try:
        if not symbol.endswith(".NS"):
            symbol = symbol + ".NS"
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        if not data.empty:
            return float(data["Close"].iloc[-1])
        return 0.0
    except Exception:
        return 0.0
