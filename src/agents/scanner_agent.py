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
        # Try multiple models in case one is unavailable
        models_to_try = [
            ("meta-llama/Meta-Llama-3-8B-Instruct", "Llama-3-8B"),
            ("google/flan-t5-large", "Flan-T5-Large"),
            ("HuggingFaceH4/zephyr-7b-beta", "Zephyr-7B")
        ]
        
        self.llm = None
        for repo_id, model_name in models_to_try:
            try:
                print(f"⏳ Trying to load {model_name}...")
                self.llm = HuggingFaceEndpoint(
                    repo_id=repo_id,
                    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
                    temperature=0.3,
                    max_new_tokens=512,
                    task="text-generation"
                )
                print(f"✅ Successfully loaded FREE model: {model_name}")
                break
            except Exception as e:
                print(f"{model_name} unavailable: {str(e)}")
                continue
        
        if not self.llm:
            raise ValueError("Could not load any HuggingFace model. Please check your token.")
        
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
            print(f"\nScanning {symbol}...")
            
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
            is_interesting = "INTERESTING: Yes" in analysis.upper() or "INTERESTING:Yes" in analysis.upper()
            
            # Always return result with reasoning
            return {
                'symbol': symbol,
                'price': latest['price'],
                'rsi': latest['rsi'],
                'analysis': analysis,
                'indicators': latest,
                'interesting': is_interesting  # Flag to indicate if picked
            }
            
            if is_interesting:
                print(f"   Interesting!")
            else:
                print(f"   Not interesting")
                
        except Exception as e:
            print(f"   Error scanning {symbol}: {str(e)}")
            return None
    
    def scan(self, symbols: List[str]) -> List[Dict]:
        """
        Scan multiple stocks and return ALL results with reasons.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            List of dicts with analysis for ALL stocks
        """
        print("\n" + "="*60)
        print(f"Starting market scan for {len(symbols)} stocks")
        print("="*60)
        
        results = []
        
        for symbol in symbols:
            result = self.scan_stock(symbol)
            if result:  # Will always be True now (we return all results)
                results.append(result)
        
        interesting_count = sum(1 for r in results if r.get('interesting', False))
        
        print("\n" + "="*60)
        print(f"Scan complete! {interesting_count} interesting, {len(results) - interesting_count} not interesting")
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
    print("Market Scanner Agent - Test Run")
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
        print("INTERESTING STOCKS FOUND:")
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
        print("\nNo interesting stocks found in this scan.")
        print("   (This is normal - not every stock has clear signals)")
    
    print("\n" + "="*60)
    print("Scanner test complete!")
    print("="*60)
