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

import pandas as pd
import numpy as np
from typing import Dict


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
        delta = df['Close'].diff()
        
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
    
    def calculate_macd(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
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
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD line
        macd_line = ema_12 - ema_26
        
        # Signal line
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        return {
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        }
    
    def calculate_bollinger_bands(
        self, 
        df: pd.DataFrame, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Dict[str, pd.Series]:
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
        middle_band = df['Close'].rolling(window=period).mean()
        
        # Standard deviation
        std = df['Close'].rolling(window=period).std()
        
        # Upper and lower bands
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        return {
            'Upper': upper_band,
            'Middle': middle_band,
            'Lower': lower_band
        }
    
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
        result['RSI'] = self.calculate_rsi(df)
        
        # Add MACD
        macd_data = self.calculate_macd(df)
        result['MACD'] = macd_data['MACD']
        result['MACD_Signal'] = macd_data['Signal']
        result['MACD_Histogram'] = macd_data['Histogram']
        
        # Add Bollinger Bands
        bb_data = self.calculate_bollinger_bands(df)
        result['BB_Upper'] = bb_data['Upper']
        result['BB_Middle'] = bb_data['Middle']
        result['BB_Lower'] = bb_data['Lower']
        
        # Add Volume Moving Average
        result['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        print("Indicators calculated!")
        
        return result
    
    def get_latest_signals(self, df_with_indicators: pd.DataFrame) -> Dict:
        """
        Get the most recent indicator values.
        
        Args:
            df_with_indicators: DataFrame after calling calculate_all()
            
        Returns:
            Dict with latest indicator values
        """
        latest = df_with_indicators.iloc[-1]
        
        return {
            'price': latest['Close'],
            'rsi': latest['RSI'],
            'macd': latest['MACD'],
            'macd_signal': latest['MACD_Signal'],
            'bb_position': self._calculate_bb_position(latest),
            'volume_ratio': latest['Volume'] / latest['Volume_MA'] if latest['Volume_MA'] > 0 else 1.0
        }
    
    def _calculate_bb_position(self, row: pd.Series) -> float:
        """Calculate where price is within Bollinger Bands."""
        if pd.isna(row['BB_Upper']) or pd.isna(row['BB_Lower']):
            return 0.5
        
        band_range = row['BB_Upper'] - row['BB_Lower']
        if band_range == 0:
            return 0.5
        
        return (row['Close'] - row['BB_Lower']) / band_range


# Test the indicators
if __name__ == "__main__":
    """
    Test script - run with: python -m src.data_pipeline.indicators_simple
    """
    from .collector import MarketDataCollector
    
    print("\n" + "="*60)
    print("Testing Simple Technical Indicators (Manual Calculation)")
    print("="*60 + "\n")
    
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
    if latest['rsi'] > 70:
        print("(Overbought)")
    elif latest['rsi'] < 30:
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
    print(data_with_indicators[['Close', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower']].tail())
    
    print("\n" + "="*60)
    print("Manual indicators working!")
    print("="*60)