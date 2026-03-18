"""
Rule-Based Scanner (No AI Required)

WHY THIS EXISTS:
- Backup if HuggingFace models are unavailable
- Uses pure logic rules instead of AI
- Still effective for technical analysis

RULES:
- Oversold: RSI < 30 + MACD positive
- Overbought: RSI > 70
- Breakout: Volume > 2x + RSI mid-range
- Neutral: Nothing significant
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


from src.data_pipeline.collector import MarketDataCollector
from src.data_pipeline.indicators_simple import SimpleTechnicalIndicators


class RuleBasedScanner:
    """
    Simple rule-based scanner without AI.
    
    Uses if/else logic to identify patterns.
    """

    def __init__(self):
        """Initialize the scanner."""
        self.collector = MarketDataCollector()
        self.indicators = SimpleTechnicalIndicators()
        print("✅ Using RULE-BASED scanner (no AI needed)")

    def analyze_indicators(self, indicators: dict) -> dict:
        """
        Analyze indicators using simple rules.
        
        Args:
            indicators: Dict with RSI, MACD, volume_ratio
            
        Returns:
            Dict with interesting flag and reasoning
        """
        rsi = indicators['rsi']
        macd = indicators['macd']
        volume_ratio = indicators['volume_ratio']

        # Rule 1: Oversold reversal
        if rsi < 30 and macd > 0:
            return {
                'interesting': True,
                'signal': 'Oversold Reversal',
                'reason': f'RSI at {rsi:.1f} (oversold) with positive MACD {macd:.2f}. Potential bounce.'
            }

        # Rule 2: Strong oversold
        if rsi < 25:
            return {
                'interesting': True,
                'signal': 'Extreme Oversold',
                'reason': f'RSI at {rsi:.1f} shows extreme selling. Mean reversion likely.'
            }

        # Rule 3: Overbought warning
        if rsi > 70:
            return {
                'interesting': True,
                'signal': 'Overbought',
                'reason': f'RSI at {rsi:.1f} indicates overbought. Potential pullback ahead.'
            }

        # Rule 4: Volume breakout
        if volume_ratio > 2.0 and 40 < rsi < 60:
            return {
                'interesting': True,
                'signal': 'Volume Breakout',
                'reason': f'Volume spike {volume_ratio:.1f}x with neutral RSI. Strong interest.'
            }

        # Rule 5: MACD momentum
        if macd > 0 and 45 < rsi < 65:
            return {
                'interesting': True,
                'signal': 'Bullish Momentum',
                'reason': f'Positive MACD {macd:.2f} with healthy RSI {rsi:.1f}. Trend is up.'
            }

        # No clear signal
        return {
            'interesting': False,
            'signal': 'Neutral',
            'reason': f'RSI at {rsi:.1f}, MACD {macd:.2f}. No clear entry/exit signal currently.'
        }

    def scan_stock(self, symbol: str) -> dict | None:
        """Scan a single stock."""
        try:
            print(f"\n🔍 Scanning {symbol}...")

            # Get data
            price_data = self.collector.fetch_data(symbol, period="3mo")
            data_with_indicators = self.indicators.calculate_all(price_data)
            latest = self.indicators.get_latest_signals(data_with_indicators)

            # Analyze using rules
            analysis = self.analyze_indicators(latest)

            # Format analysis text
            analysis_text = f"""INTERESTING: {'Yes' if analysis['interesting'] else 'No'}
SIGNAL: {analysis['signal']}
REASON: {analysis['reason']}"""

            if analysis['interesting']:
                print("   ✅ Interesting!")
            else:
                print("   ⏭️  Not interesting")

            return {
                'symbol': symbol,
                'price': latest['price'],
                'rsi': latest['rsi'],
                'analysis': analysis_text,
                'indicators': latest,
                'interesting': analysis['interesting']
            }

        except Exception as e:
            print(f"   ❌ Error scanning {symbol}: {str(e)}")
            return None

    def scan(self, symbols: list[str]) -> list[dict]:
        """Scan multiple stocks."""
        print("\n" + "="*60)
        print(f"🚀 Starting rule-based scan for {len(symbols)} stocks")
        print("="*60)

        results = []
        for symbol in symbols:
            result = self.scan_stock(symbol)
            if result:
                results.append(result)

        interesting_count = sum(1 for r in results if r.get('interesting', False))

        print("\n" + "="*60)
        print(f"✅ Scan complete! {interesting_count} interesting, {len(results) - interesting_count} not interesting")
        print("="*60)

        return results


# For testing
if __name__ == "__main__":
    from src.agents.scanner_agent import NIFTY_50_SAMPLE

    print("\n" + "="*60)
    print("🤖 Rule-Based Scanner - Test Run")
    print("="*60)

    scanner = RuleBasedScanner()
    test_symbols = NIFTY_50_SAMPLE[:3]

    results = scanner.scan(test_symbols)

    if results:
        print("\n📊 RESULTS:")
        for r in results:
            print(f"\n{r['symbol']}: {'✅ INTERESTING' if r['interesting'] else '⏭️  Not interesting'}")
            print(r['analysis'])
