"""
Market Scanner Agent

WHY THIS EXISTS:
- Automate the process of finding trading opportunities
- Use AI to identify interesting patterns in indicators
- Provide human-readable explanations for findings

WHAT IT DOES:
1. Takes a list of stocks to scan
2. Fetches data and calculates indicators for each
3. Uses Claude AI to analyze the indicators
4. Returns stocks that look interesting with explanations

HOW TO USE:
    scanner = MarketScanner()
    results = scanner.scan(['RELIANCE.NS', 'TCS.NS', 'INFY.NS'])
    for result in results:
        print(f"{result['symbol']}: {result['reasoning']}")
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from typing import List, Dict
from src.data_pipeline.collector import MarketDataCollector
from src.data_pipeline.indicators import SimpleTechnicalIndicators
from src.utils.config import HUGGINGFACE_API_TOKEN
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate


class MarketScanner:
    """
    Scans stocks and identifies trading opportunities using AI.
    
    This is our first AI agent! It uses Claude to analyze
    technical indicators and explain opportunities.
    """
    
    def __init__(self):
        """Initialize the scanner with data collector and AI model."""
        self.collector = MarketDataCollector()
        self.indicators = SimpleTechnicalIndicators()
        
        # Initialize HuggingFace LLM (FREE!)
        # Using Mistral-7B-Instruct - good for analysis, completely free
        self.llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # Free, open-source model
            huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
            temperature=0.3,  # Lower = more focused
            max_new_tokens=512  # Max response length
        )
        
        print("✅ Using FREE HuggingFace model: Mistral-7B-Instruct")
        
        # Create the analysis prompt
        # Simpler prompt for open-source models
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a technical analyst. Analyze stock indicators and identify trading opportunities.

Look for these signals:
- RSI < 30 = Oversold (potential buy)
- RSI > 70 = Overbought (potential sell)
- MACD positive + RSI mid-range = Bullish
- Volume > 2x average = Strong interest

Format your response EXACTLY like this:
INTERESTING: Yes or No
SIGNAL: (one word: Oversold, Overbought, Breakout, or Neutral)
REASON: (one sentence explaining why)
"""),
            ("user", """Stock: {symbol}
Price: {price}
RSI: {rsi}
MACD: {macd}
Volume Ratio: {volume_ratio}x

Analyze:""")
        ])
    
    def scan_stock(self, symbol: str) -> Dict | None:
        """
        Scan a single stock for trading opportunities.
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE.NS")
            
        Returns:
            Dict with analysis if interesting, None otherwise
        """
        try:
            print(f"\n🔍 Scanning {symbol}...")
            
            # Step 1: Get price data (3 months for good indicator calculation)
            price_data = self.collector.fetch_data(symbol, period="3mo")
            
            # Step 2: Calculate indicators
            data_with_indicators = self.indicators.calculate_all(price_data)
            
            # Step 3: Get latest values
            latest = self.indicators.get_latest_signals(data_with_indicators)
            
            # Step 4: Ask the AI to analyze (simplified for open-source model)
            chain = self.analysis_prompt | self.llm
            response = chain.invoke({
                "symbol": symbol,
                "price": f"{latest['price']:.2f}",
                "rsi": f"{latest['rsi']:.2f}",
                "macd": f"{latest['macd']:.2f}",
                "volume_ratio": f"{latest['volume_ratio']:.2f}"
            })
            
            # Step 5: Parse the model's response
            analysis = response
            
            # Check if the model found it interesting
            if "INTERESTING: Yes" in analysis.upper() or "INTERESTING:Yes" in analysis.upper():
                return {
                    'symbol': symbol,
                    'price': latest['price'],
                    'rsi': latest['rsi'],
                    'analysis': analysis,
                    'indicators': latest
                }
            else:
                print(f"   ⏭️  Not interesting - skipping")
                return None
                
        except Exception as e:
            print(f"   ❌ Error scanning {symbol}: {str(e)}")
            return None
    
    def scan(self, symbols: List[str]) -> List[Dict]:
        """
        Scan multiple stocks and return interesting ones.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            List of dicts with analysis for interesting stocks
        """
        print("\n" + "="*60)
        print(f"🚀 Starting market scan for {len(symbols)} stocks")
        print("="*60)
        
        results = []
        
        for symbol in symbols:
            result = self.scan_stock(symbol)
            if result:
                results.append(result)
        
        print("\n" + "="*60)
        print(f"✅ Scan complete! Found {len(results)} interesting stocks")
        print("="*60)
        
        return results


# Nifty 50 stocks (sample list)
NIFTY_50_SAMPLE = [
    'RELIANCE.NS',
    'TCS.NS',
    'HDFCBANK.NS',
    'INFY.NS',
    'ICICIBANK.NS',
    'HINDUNILVR.NS',
    'ITC.NS',
    'SBIN.NS',
    'BHARTIARTL.NS',
    'KOTAKBANK.NS'
]


# Test the scanner
if __name__ == "__main__":
    """
    Test script - run with: python src/agents/scanner_agent.py
    """
    
    print("\n" + "="*60)
    print("🤖 Market Scanner Agent - Test Run")
    print("="*60)
    
    # Create scanner
    scanner = MarketScanner()
    
    # Scan a small sample (to save API calls)
    test_symbols = NIFTY_50_SAMPLE[:3]  # Just 3 stocks for testing
    
    print(f"\nTesting with: {', '.join(test_symbols)}")
    
    # Run scan
    results = scanner.scan(test_symbols)
    
    # Display results
    if results:
        print("\n" + "="*60)
        print("📊 INTERESTING STOCKS FOUND:")
        print("="*60)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['symbol']}")
            print(f"   Price: ₹{result['price']:.2f}")
            print(f"   RSI: {result['rsi']:.2f}")
            print("\n   AI Analysis:")
            print("   " + "-"*56)
            # Print analysis indented
            for line in result['analysis'].split('\n'):
                if line.strip():
                    print(f"   {line}")
            print()
    else:
        print("\n📭 No interesting stocks found in this scan.")
        print("   (This is normal - not every stock has clear signals)")
    
    print("\n" + "="*60)
    print("✅ Scanner test complete!")
    print("="*60)
