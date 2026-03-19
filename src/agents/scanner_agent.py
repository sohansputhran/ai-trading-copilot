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

from huggingface_hub import InferenceClient

from src.data_pipeline.collector import MarketDataCollector
from src.data_pipeline.indicators import SimpleTechnicalIndicators
from src.utils.config import HUGGINGFACE_API_TOKEN


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

        # Initialize HuggingFace InferenceClient (FREE!)
        # Uses the new router.huggingface.co API via chat_completion()
        # Try multiple models in case one is unavailable
        models_to_try = [
            ("meta-llama/Meta-Llama-3-8B-Instruct", "Llama-3-8B"),  # Confirmed working
            ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B"),
            ("microsoft/Phi-3-mini-4k-instruct", "Phi-3-mini"),
            ("HuggingFaceH4/zephyr-7b-beta", "Zephyr-7B"),
        ]

        self.llm_model = None
        self.llm_client = None

        for repo_id, model_name in models_to_try:
            try:
                print(f"⏳ Trying to load {model_name}...")
                client = InferenceClient(model=repo_id, token=HUGGINGFACE_API_TOKEN, timeout=30)
                # Quick connectivity test
                test_resp = client.chat_completion(
                    messages=[{"role": "user", "content": "Reply with just: OK"}], max_tokens=5
                )
                _ = test_resp.choices[0].message.content
                self.llm_client = client
                self.llm_model = model_name
                print(f"✅ Successfully connected to FREE model: {model_name}")
                break
            except Exception as e:
                print(f"   {model_name} unavailable: {str(e)[:120]}")
                continue

        if not self.llm_client:
            print("⚠️  No AI model available — will use rule-based fallback for all scans.")

        # System prompt - strict about when NOT to flag as interesting
        self.system_prompt = (
            "You are a strict technical stock screener. Your job is to filter OUT most stocks. "
            "Only mark a stock as INTERESTING if it meets AT LEAST ONE of these exact conditions:\n"
            "  - RSI below 30 (oversold)\n"
            "  - RSI above 70 (overbought)\n"
            "  - Volume ratio above 2.0x AND RSI between 40-60\n"
            "  - MACD above 5.0 AND RSI between 45-65\n"
            "If NONE of these conditions are met, you MUST say INTERESTING: No. "
            "Most stocks (around 70-80%) should be NOT interesting. Be strict and conservative. "
            "Format your response EXACTLY like this (3 lines, nothing else):\n"
            "INTERESTING: Yes or No\n"
            "SIGNAL: Oversold, Overbought, Breakout, Bullish, or Neutral\n"
            "REASON: one sentence with the specific indicator values that support your decision"
        )

    def scan_stock(self, symbol: str) -> dict | None:
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

            # Step 4: Try AI analysis, fall back to rules if it fails
            try:
                if not self.llm_client:
                    raise RuntimeError("No AI client available")

                rsi = latest["rsi"]
                macd = latest["macd"]
                vol = latest["volume_ratio"]

                # Pre-compute whether ANY trigger condition is met
                triggers = []
                if rsi < 30:
                    triggers.append(f"RSI={rsi:.1f} is OVERSOLD (<30)")
                if rsi > 70:
                    triggers.append(f"RSI={rsi:.1f} is OVERBOUGHT (>70)")
                if vol > 2.0 and 40 < rsi < 60:
                    triggers.append(f"Volume={vol:.1f}x is HIGH with neutral RSI")
                if macd > 5.0 and 45 < rsi < 65:
                    triggers.append(f"MACD={macd:.1f} is STRONG with healthy RSI")
                hint = f"Pre-analysis: {', '.join(triggers) if triggers else 'NO trigger conditions met — likely Neutral'}."

                user_message = (
                    f"Stock: {symbol}\n"
                    f"Price: {latest['price']:.2f}\n"
                    f"RSI: {rsi:.2f}\n"
                    f"MACD: {macd:.2f}\n"
                    f"Volume Ratio: {vol:.2f}x\n"
                    f"{hint}\n\n"
                    f"Based on the criteria above, is this stock interesting?"
                )

                response = self.llm_client.chat_completion(
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    max_tokens=200,
                    temperature=0.3,
                )

                # Step 5: Parse the model's response
                analysis = response.choices[0].message.content.strip()
                is_interesting = (
                    "INTERESTING: YES" in analysis.upper() or "INTERESTING:YES" in analysis.upper()
                )
                print(f"   AI analysis complete ({self.llm_model})")

            except Exception as ai_error:
                # AI failed - use rule-based fallback
                print(f"   AI error: {type(ai_error).__name__}: {str(ai_error)[:150]}")
                print("   Falling back to rule-based analysis")
                is_interesting, analysis = self._rule_based_fallback(latest)

            # Always return result with reasoning
            return {
                "symbol": symbol,
                "price": latest["price"],
                "rsi": latest["rsi"],
                "analysis": analysis,
                "indicators": latest,
                "interesting": is_interesting,  # Flag to indicate if picked
            }

        except Exception as e:
            print(f"   Error scanning {symbol}: {str(e)}")
            return None

    def _rule_based_fallback(self, indicators: dict) -> tuple:
        """
        Rule-based analysis when AI fails.

        Args:
            indicators: Dict with RSI, MACD, volume_ratio

        Returns:
            Tuple of (is_interesting: bool, analysis: str)
        """
        rsi = indicators["rsi"]
        macd = indicators["macd"]
        volume_ratio = indicators["volume_ratio"]

        # Rule 1: Oversold reversal
        if rsi < 30 and macd > 0:
            return (
                True,
                f"""INTERESTING: Yes
SIGNAL: Oversold Reversal
REASON: RSI {rsi:.1f} (oversold) + positive MACD {macd:.2f} suggests potential bounce.""",
            )

        # Rule 2: Extreme oversold
        if rsi < 25:
            return (
                True,
                f"""INTERESTING: Yes
SIGNAL: Extreme Oversold
REASON: RSI {rsi:.1f} shows heavy selling. Mean reversion likely.""",
            )

        # Rule 3: Overbought
        if rsi > 70:
            return (
                True,
                f"""INTERESTING: Yes
SIGNAL: Overbought
REASON: RSI {rsi:.1f} indicates extended run. Pullback possible.""",
            )

        # Rule 4: Volume breakout
        if volume_ratio > 2.0 and 40 < rsi < 60:
            return (
                True,
                f"""INTERESTING: Yes
SIGNAL: Volume Breakout
REASON: Volume {volume_ratio:.1f}x average with neutral RSI shows strong interest.""",
            )

        # Rule 5: Bullish momentum
        if macd > 5 and 45 < rsi < 65:
            return (
                True,
                f"""INTERESTING: Yes
SIGNAL: Bullish Momentum
REASON: Positive MACD {macd:.2f} + healthy RSI {rsi:.1f} = uptrend.""",
            )

        # No clear signal
        return (
            False,
            f"""INTERESTING: No
SIGNAL: Neutral
REASON: RSI {rsi:.1f}, MACD {macd:.2f}, Volume {volume_ratio:.1f}x. No clear setup.""",
        )

    def scan(self, symbols: list[str]) -> list[dict]:
        """
        Scan multiple stocks and return ALL results with reasons.

        Args:
            symbols: List of stock symbols

        Returns:
            List of dicts with analysis for ALL stocks
        """
        print("\n" + "=" * 60)
        print(f"Starting market scan for {len(symbols)} stocks")
        print("=" * 60)

        results = []

        for symbol in symbols:
            result = self.scan_stock(symbol)
            if result:  # Will always be True now (we return all results)
                results.append(result)

        interesting_count = sum(1 for r in results if r.get("interesting", False))

        print("\n" + "=" * 60)
        print(
            f"Scan complete! {interesting_count} interesting, {len(results) - interesting_count} not interesting"
        )
        print("=" * 60)

        return results


# Nifty 50 stocks (sample list)
NIFTY_50_SAMPLE: list[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "KOTAKBANK.NS",
]


# Test the scanner
if __name__ == "__main__":
    """
    Test script - run with: python src/agents/scanner_agent.py
    """

    print("\n" + "=" * 60)
    print("Market Scanner Agent - Test Run")
    print("=" * 60)

    # Create scanner
    scanner = MarketScanner()

    # Scan a small sample (to save API calls)
    test_symbols = NIFTY_50_SAMPLE[:3]  # Just 3 stocks for testing

    print(f"\nTesting with: {', '.join(test_symbols)}")

    # Run scan
    results = scanner.scan(test_symbols)

    # Display results
    if results:
        print("\n" + "=" * 60)
        print("INTERESTING STOCKS FOUND:")
        print("=" * 60)

        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['symbol']}")
            print(f"   Price: {result['price']:.2f}")
            print(f"   RSI: {result['rsi']:.2f}")
            print("\n   AI Analysis:")
            print("   " + "-" * 56)
            # Print analysis indented
            for line in result["analysis"].split("\n"):
                if line.strip():
                    print(f"   {line}")
            print()
    else:
        print("\nNo interesting stocks found in this scan.")
        print("   (This is normal - not every stock has clear signals)")

    print("\n" + "=" * 60)
    print("Scanner test complete!")
    print("=" * 60)
