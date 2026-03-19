"""
Simple Technical Indicators (Manual Calculation)

WHY THIS VERSION?
- No dependency issues
- Works with just pandas and numpy
- Educational - you see exactly how indicators are calculated
- Use this if ta library doesn't install

TRADE-OFF:
- More code than using a library
- But you understand what's happening!
"""

import numpy as np
import pandas as pd


class SimpleTechnicalIndicators:
    """
    Calculate technical indicators manually using just pandas.

    No external indicator libraries needed - just math!
    """

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate RSI manually.

        HOW RSI IS CALCULATED:
        1. Calculate price changes (today - yesterday)
        2. Separate into gains and losses
        3. Average the gains and losses over 14 days
        4. RS = Average Gain / Average Loss
        5. RSI = 100 - (100 / (1 + RS))

        Args:
            df: DataFrame with 'Close' column
            period: Lookback period (default 14)

        Returns:
            Series with RSI values
        """
        # Step 1: Calculate price changes
        delta = df["Close"].diff()

        # Step 2: Separate gains and losses
        gain = delta.where(delta > 0, 0)  # Keep gains, set losses to 0
        loss = -delta.where(delta < 0, 0)  # Keep losses (as positive), set gains to 0

        # Step 3: Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # Step 4: Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss

        # Step 5: Calculate RSI
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """
        Calculate MACD manually.

        HOW MACD IS CALCULATED:
        1. Calculate 12-day EMA (fast)
        2. Calculate 26-day EMA (slow)
        3. MACD = Fast EMA - Slow EMA
        4. Signal = 9-day EMA of MACD
        5. Histogram = MACD - Signal

        EMA = Exponential Moving Average (gives more weight to recent prices)

        Args:
            df: DataFrame with 'Close' column

        Returns:
            Dict with 'MACD', 'Signal', 'Histogram'
        """
        # Calculate EMAs
        ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["Close"].ewm(span=26, adjust=False).mean()

        # MACD line
        macd_line = ema_12 - ema_26

        # Signal line
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        # Histogram
        histogram = macd_line - signal_line

        return {"MACD": macd_line, "Signal": signal_line, "Histogram": histogram}

    def calculate_bollinger_bands(
        self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
    ) -> dict[str, pd.Series]:
        """
        Calculate Bollinger Bands manually.

        HOW BOLLINGER BANDS ARE CALCULATED:
        1. Middle Band = 20-day Simple Moving Average
        2. Standard Deviation = Measure of price volatility
        3. Upper Band = Middle + (2 × Std Dev)
        4. Lower Band = Middle - (2 × Std Dev)

        Args:
            df: DataFrame with 'Close' column
            period: Lookback period (default 20)
            std_dev: Number of standard deviations (default 2.0)

        Returns:
            Dict with 'Upper', 'Middle', 'Lower' bands
        """
        # Middle band (SMA)
        middle_band = df["Close"].rolling(window=period).mean()

        # Standard deviation
        std = df["Close"].rolling(window=period).std()

        # Upper and lower bands
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)

        return {"Upper": upper_band, "Middle": middle_band, "Lower": lower_band}

    def calculate_ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.

        HOW EMA IS CALCULATED:
        - Like a simple moving average, but recent prices are weighted more heavily
        - Multiplier = 2 / (period + 1)
        - EMA = (Close - Previous EMA) × Multiplier + Previous EMA
        - pandas ewm(span=period) handles this efficiently

        WHY WE NEED TWO EMAs (20 and 50):
        - EMA20 reacts faster (short-term trend)
        - EMA50 reacts slower (medium-term trend)
        - When EMA20 crosses above EMA50 → bullish signal (golden cross)
        - When EMA20 crosses below EMA50 → bearish signal (death cross)

        Args:
            df: DataFrame with 'Close' column
            period: EMA period (20 or 50)

        Returns:
            Series with EMA values
        """
        return df["Close"].ewm(span=period, adjust=False).mean()

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate ADX (Average Directional Index).

        HOW ADX IS CALCULATED:
        1. Calculate True Range (TR) — largest of:
           - High - Low
           - |High - Previous Close|
           - |Low - Previous Close|
        2. Calculate +DM (positive directional movement) and -DM (negative)
        3. Smooth both over 'period' days
        4. DI+ = 100 × Smoothed +DM / ATR
           DI- = 100 × Smoothed -DM / ATR
        5. DX = 100 × |DI+ - DI-| / (DI+ + DI-)
        6. ADX = Smoothed average of DX

        WHY ADX MATTERS FOR MOMENTUM AGENT:
        - ADX measures trend STRENGTH, not direction
        - ADX < 20: no clear trend (choppy market, avoid momentum trades)
        - ADX 20–40: developing trend
        - ADX > 40: strong trend (high conviction for momentum agent)
        - ADX doesn't tell you which direction — DI+ vs DI- does that

        Args:
            df: DataFrame with 'High', 'Low', 'Close' columns
            period: Smoothing period (default 14)

        Returns:
            Series with ADX values
        """
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)

        # Smooth over period
        atr_smooth = tr.ewm(span=period, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(span=period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(span=period, adjust=False).mean()

        # Directional indicators
        plus_di = 100 * plus_dm_smooth / atr_smooth
        minus_di = 100 * minus_dm_smooth / atr_smooth

        # DX and ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate ATR (Average True Range).

        HOW ATR IS CALCULATED:
        1. True Range = max of:
           - High - Low (today's range)
           - |High - Yesterday's Close| (gap up)
           - |Low  - Yesterday's Close| (gap down)
        2. ATR = 14-day EMA of True Range

        WHY ATR MATTERS FOR BREAKOUT AGENT:
        - ATR measures volatility in price units (e.g. ₹35 per day)
        - A breakout is only meaningful if the price move exceeds normal volatility
        - If ATR is ₹20 and price moves ₹15 — that's noise, not a breakout
        - If ATR is ₹20 and price moves ₹40 — that's a real breakout
        - We use: price_move > 1.5 × ATR as the breakout confirmation threshold

        Args:
            df: DataFrame with 'High', 'Low', 'Close' columns
            period: Smoothing period (default 14)

        Returns:
            Series with ATR values
        """
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return tr.ewm(span=period, adjust=False).mean()

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators at once.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with original data + indicator columns
        """
        result = df.copy()

        print("Calculating technical indicators (manual calculation)...")

        # Add RSI
        result["RSI"] = self.calculate_rsi(df)

        # Add MACD
        macd_data = self.calculate_macd(df)
        result["MACD"] = macd_data["MACD"]
        result["MACD_Signal"] = macd_data["Signal"]
        result["MACD_Histogram"] = macd_data["Histogram"]

        # Add Bollinger Bands
        bb_data = self.calculate_bollinger_bands(df)
        result["BB_Upper"] = bb_data["Upper"]
        result["BB_Middle"] = bb_data["Middle"]
        result["BB_Lower"] = bb_data["Lower"]

        # Add Volume Moving Average
        result["Volume_MA"] = df["Volume"].rolling(window=20).mean()

        # EMAs (for Momentum Agent)
        result["EMA_20"] = self.calculate_ema(df, period=20)
        result["EMA_50"] = self.calculate_ema(df, period=50)

        # EMA cross: +1 = EMA20 above EMA50 (bullish), -1 = below (bearish), 0 = equal
        result["EMA_Cross"] = np.sign(result["EMA_20"] - result["EMA_50"])

        # ADX — trend strength (for Momentum Agent)
        result["ADX"] = self.calculate_adx(df)

        # ATR — volatility in price units (for Breakout Agent)
        result["ATR"] = self.calculate_atr(df)

        # Resistance — highest high over last 20 days (for Breakout Agent)
        # This is a simple but effective resistance proxy
        result["Resistance"] = df["High"].rolling(window=20).max()

        print("Indicators calculated!")

        return result

    def get_latest_signals(self, df_with_indicators: pd.DataFrame) -> dict:
        """
        Get the most recent indicator values as a flat dict.

        This dict is passed directly into TradingState["indicators"] and
        read by all Sprint 2 agents. Key names here must match what agents
        call via safe_get().

        Sprint 1 keys (unchanged):
            price, rsi, macd, macd_signal, bb_position, volume_ratio

        Sprint 2 additions:
            ema_20, ema_50, ema_cross, adx, atr,
            resistance, price_vs_resistance
        """
        latest = df_with_indicators.iloc[-1]

        # Price vs resistance: 0.0 = far below, 1.0 = exactly at resistance, >1.0 = breakout
        resistance = latest["Resistance"] if not pd.isna(latest["Resistance"]) else latest["Close"]
        price_vs_resistance = latest["Close"] / resistance if resistance > 0 else 1.0

        return {
            "price": latest["Close"],
            "rsi": latest["RSI"],
            "macd": latest["MACD"],
            "macd_signal": latest["MACD_Signal"],
            "bb_position": self._calculate_bb_position(latest),
            "volume_ratio": (
                latest["Volume"] / latest["Volume_MA"] if latest["Volume_MA"] > 0 else 1.0
            ),
            # For Momentum Agent
            "ema_20": latest["EMA_20"],
            "ema_50": latest["EMA_50"],
            "ema_cross": latest["EMA_Cross"],  # +1 bullish, -1 bearish
            "adx": latest["ADX"],  # >20 = trending, >40 = strong trend
            # For Breakout Agent
            "atr": latest["ATR"],
            "resistance": resistance,
            "price_vs_resistance": price_vs_resistance,  # >0.98 = near breakout
        }

    def _calculate_bb_position(self, row: pd.Series) -> float:
        """Calculate where price is within Bollinger Bands."""
        if pd.isna(row["BB_Upper"]) or pd.isna(row["BB_Lower"]):
            return 0.5

        band_range = row["BB_Upper"] - row["BB_Lower"]
        if band_range == 0:
            return 0.5

        return (row["Close"] - row["BB_Lower"]) / band_range


# Test the indicators
if __name__ == "__main__":
    """
    Test script - run with: python -m src.data_pipeline.indicators_simple
    """
    from .collector import MarketDataCollector

    print("\n" + "=" * 60)
    print("Testing Simple Technical Indicators (Manual Calculation)")
    print("=" * 60 + "\n")

    # Get price data
    print("Fetching price data...")
    collector = MarketDataCollector()
    price_data = collector.fetch_data("RELIANCE.NS", period="3mo")

    # Calculate indicators
    print("\nCalculating indicators...")
    calculator = SimpleTechnicalIndicators()
    data_with_indicators = calculator.calculate_all(price_data)

    # Show latest values
    print("\nLatest indicator values:")
    print("-" * 60)
    latest = calculator.get_latest_signals(data_with_indicators)

    print(f"Price: ₹{latest['price']:.2f}")
    print(f"RSI: {latest['rsi']:.2f} ", end="")
    if latest["rsi"] > 70:
        print("(Overbought)")
    elif latest["rsi"] < 30:
        print("(Oversold)")
    else:
        print("(Neutral)")

    print(f"MACD: {latest['macd']:.2f}")
    print(f"MACD Signal: {latest['macd_signal']:.2f}")
    print(f"Bollinger Band Position: {latest['bb_position']:.2f}")
    print(f"Volume Ratio: {latest['volume_ratio']:.2f}x average")

    # Show sample data
    print("\n" + "-" * 60)
    print("Sample data (last 5 days):")
    print(data_with_indicators[["Close", "RSI", "MACD", "BB_Upper", "BB_Lower"]].tail())

    print("\n" + "=" * 60)
    print("Manual indicators working!")
    print("=" * 60)
