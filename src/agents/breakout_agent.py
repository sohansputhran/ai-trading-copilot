"""
Breakout Strategy Agent
========================
Specializes in detecting price breakouts confirmed by volume expansion.

This agent answers one question:
    "Is price breaking through a significant level with enough volume to confirm it?"

WHY BREAKOUT AS A SEPARATE AGENT?
Breakouts are high-risk, high-reward setups that require different conditions
than momentum or oscillator signals. A breakout trade needs:
    1. Price approaching or clearing a key level (resistance or support)
    2. Volume significantly above average (smart money participation)
    3. Volatility expansion (ATR confirms the move is real, not noise)

A stock can have a neutral RSI (Technical Agent says HOLD) and no EMA cross
(Momentum Agent says HOLD), yet still be on the verge of a major breakout.
This agent catches those situations the other two would miss entirely.

Indicators used (all from get_latest_signals()):
    'volume_ratio'         → current volume / 20-day avg volume
    'price_vs_resistance'  → price / 20-day high (1.0 = at resistance)
    'atr'                  → average true range (daily volatility in ₹)
    'price'                → current close price
    'resistance'           → 20-day high (resistance level)
    'rsi'                  → filter only (breakouts from extreme RSI are risky)

Scoring logic (max 5.0 pts):
    Volume confirmation:        2.0 pts  (is smart money participating?)
    Price vs resistance:        2.0 pts  (how close/past the breakout level?)
    ATR expansion:              1.0 pts  (is volatility supporting the move?)
"""

from typing import Any

import structlog

from src.agents.base_agent import BaseStrategyAgent
from src.agents.state import AgentAnalysis, AgentName, Signal, TradingState

logger = structlog.get_logger()


class BreakoutStrategyAgent(BaseStrategyAgent):
    """
    Volume + price breakout detection agent.
    Combines rule-based signals with optional LLM reasoning enrichment.
    """

    agent_name = AgentName.BREAKOUT.value

    # ── Thresholds ────────────────────────────────────────────────────────────
    VOLUME_MODERATE   = 1.5    # 1.5x avg volume — moderate interest
    VOLUME_STRONG     = 2.0    # 2.0x avg volume — strong confirmation
    VOLUME_EXTREME    = 3.0    # 3.0x avg volume — climactic (potential exhaustion)

    NEAR_RESISTANCE   = 0.97   # Within 3% of resistance = approaching breakout
    AT_RESISTANCE     = 0.99   # Within 1% = at resistance
    ABOVE_RESISTANCE  = 1.01   # 1% above = confirmed breakout

    RSI_BREAKOUT_MAX  = 80.0   # Avoid breakout buys when RSI already extreme
    RSI_BREAKDOWN_MIN = 20.0   # Avoid breakdown sells when RSI already extreme

    SYSTEM_PROMPT = (
        "You are a breakout trader specializing in volume-confirmed price breakouts. "
        "Explain in 2 sentences why the volume and price-vs-resistance values support the stated signal. "
        "Be specific: mention the volume ratio and what it tells us about market participation. "
        "Do NOT recommend actual trades. Do NOT repeat the signal word at the start."
    )

    def analyze(self, state: TradingState) -> AgentAnalysis:
        indicators = state["indicators"]
        symbol     = state["symbol"]
        self.log   = logger.bind(agent=self.agent_name, symbol=symbol)

        snapshot                              = self._extract_indicators(indicators)
        signal, confidence, rule_text, warnings = self._compute_signal(snapshot)
        reasoning = self._get_llm_reasoning(symbol, snapshot, signal) or rule_text

        return AgentAnalysis(
            agent_name    = self.agent_name,
            signal        = signal,
            confidence    = confidence,
            reasoning     = reasoning,
            key_indicators= snapshot,
            warnings      = warnings,
        )

    # ─────────────────────────────────────────────
    # Private: indicator extraction
    # ─────────────────────────────────────────────

    def _extract_indicators(self, indicators: dict[str, Any]) -> dict[str, float]:
        """Pull only the indicators this agent uses."""
        return {
            "volume_ratio":        self.safe_get(indicators, "volume_ratio",        1.0),
            "price_vs_resistance": self.safe_get(indicators, "price_vs_resistance", 0.9),
            "atr":                 self.safe_get(indicators, "atr",                  0.0),
            "price":               self.safe_get(indicators, "price",                0.0),
            "resistance":          self.safe_get(indicators, "resistance",            0.0),
            "rsi":                 self.safe_get(indicators, "rsi",                  50.0),
        }

    # ─────────────────────────────────────────────
    # Private: rule-based signal logic
    # ─────────────────────────────────────────────

    def _compute_signal(
        self, snap: dict[str, float]
    ) -> tuple[Signal, float, str, list[str]]:
        """
        Breakout scoring.

        Key insight: volume is the gate for breakouts. Price touching resistance
        on low volume is almost always a false breakout — institutions aren't
        participating. We require at least 1.5x average volume before scoring
        any bullish breakout points.

        We also detect breakdowns (price falling through support-equivalent
        levels with high volume) for SELL signals.

        Scoring:
            Volume confirmation:    2.0 pts  — participation quality
            Price vs resistance:    2.0 pts  — proximity to/above breakout level
            ATR expansion:          1.0 pts  — volatility confirming the move
        """
        vol_ratio   = snap["volume_ratio"]
        pvr         = snap["price_vs_resistance"]   # price / resistance
        rsi         = snap["rsi"]
        atr         = snap["atr"]
        price       = snap["price"]
        resistance  = snap["resistance"]

        bull_score = 0.0
        bear_score = 0.0
        reasons:  list[str] = []
        warnings: list[str] = []

        # ── Volume scoring (2.0 pts) ──────────────────────────────────────────
        if vol_ratio >= self.VOLUME_EXTREME:
            vol_pts = 2.0
            reasons.append(f"Volume {vol_ratio:.1f}x average (very strong — climactic)")
            warnings.append(
                f"Volume {vol_ratio:.1f}x is extreme — could signal climactic exhaustion, "
                f"not continuation"
            )
        elif vol_ratio >= self.VOLUME_STRONG:
            vol_pts = 2.0
            reasons.append(f"Volume {vol_ratio:.1f}x average (strong confirmation)")
        elif vol_ratio >= self.VOLUME_MODERATE:
            vol_pts = 1.0
            reasons.append(f"Volume {vol_ratio:.1f}x average (moderate interest)")
        else:
            vol_pts = 0.0
            reasons.append(f"Volume {vol_ratio:.1f}x average (insufficient for breakout)")

        # ── Price vs resistance (2.0 pts) ─────────────────────────────────────
        if pvr >= self.ABOVE_RESISTANCE:
            # Price has broken above resistance — confirmed breakout
            price_pts = 2.0
            reasons.append(
                f"Price ({price:.1f}) above resistance ({resistance:.1f}) — breakout confirmed"
            )
        elif pvr >= self.AT_RESISTANCE:
            # Price at resistance — high-probability setup, not yet confirmed
            price_pts = 1.5
            reasons.append(
                f"Price ({price:.1f}) at resistance ({resistance:.1f}) — breakout imminent"
            )
        elif pvr >= self.NEAR_RESISTANCE:
            # Approaching resistance — early warning
            price_pts = 0.8
            reasons.append(
                f"Price ({price:.1f}) approaching resistance ({resistance:.1f}) — "
                f"within {(1 - pvr) * 100:.1f}%"
            )
        else:
            # Far from resistance — no breakout setup
            price_pts = 0.0

        # Apply volume gate: no volume = no breakout points regardless of price
        if vol_ratio >= self.VOLUME_MODERATE:
            bull_score += vol_pts + price_pts
        else:
            # Low volume even at resistance is a warning, not a signal
            if pvr >= self.NEAR_RESISTANCE:
                warnings.append(
                    f"Price near resistance but volume only {vol_ratio:.1f}x — "
                    f"low conviction, likely false breakout"
                )

        # ── Breakdown detection (SELL signal) ─────────────────────────────────
        # A breakdown is the mirror: price making new lows with high volume
        # We use pvr < 0.85 (price well below its 20-day high) as a proxy
        if pvr < 0.85 and vol_ratio >= self.VOLUME_STRONG:
            bear_score += vol_pts
            bear_score += 1.5   # Price well below recent high with high volume
            reasons.append(
                f"Price ({price:.1f}) significantly below recent high with "
                f"volume {vol_ratio:.1f}x — potential breakdown"
            )

        # ── ATR expansion (1.0 pts) ───────────────────────────────────────────
        # A real breakout should have price movement > 1× ATR
        if atr > 0 and resistance > 0:
            price_move = abs(price - resistance)
            if price_move > atr:
                if bull_score > bear_score:
                    bull_score += 1.0
                else:
                    bear_score += 1.0
                reasons.append(f"Price move ({price_move:.1f}) exceeds ATR ({atr:.1f}) — real move")
            else:
                warnings.append(
                    f"Price move ({price_move:.1f}) smaller than ATR ({atr:.1f}) — "
                    f"may be noise"
                )

        # ── RSI filter ────────────────────────────────────────────────────────
        if bull_score > bear_score and rsi > self.RSI_BREAKOUT_MAX:
            warnings.append(
                f"RSI={rsi:.1f} extremely high — breakout may be exhausted"
            )
        if bear_score > bull_score and rsi < self.RSI_BREAKDOWN_MIN:
            warnings.append(
                f"RSI={rsi:.1f} extremely low — breakdown may be exhausted"
            )

        # ── Determine signal + confidence ─────────────────────────────────────
        max_possible = 5.0

        if bull_score > bear_score and bull_score > 0:
            signal     = Signal.BUY
            confidence = round(min(bull_score / max_possible, 1.0), 3)
            rule_text  = (
                f"Bullish breakout signal. {'; '.join(reasons)}. "
                f"Score: {bull_score:.1f}/{max_possible}"
            )
        elif bear_score > bull_score and bear_score > 0:
            signal     = Signal.SELL
            confidence = round(min(bear_score / max_possible, 1.0), 3)
            rule_text  = (
                f"Bearish breakdown signal. {'; '.join(reasons)}. "
                f"Score: {bear_score:.1f}/{max_possible}"
            )
        else:
            signal     = Signal.HOLD
            confidence = 0.2
            rule_text  = (
                f"No breakout setup — volume {vol_ratio:.1f}x, "
                f"price at {pvr:.0%} of resistance. Waiting for setup."
            )

        self.log.debug(
            "rule_signal_computed",
            signal=signal, bull_score=bull_score,
            bear_score=bear_score, confidence=confidence,
            volume_ratio=vol_ratio, price_vs_resistance=pvr,
        )

        return signal, confidence, rule_text, warnings

    # ─────────────────────────────────────────────
    # Private: LLM reasoning enrichment
    # ─────────────────────────────────────────────

    def _get_llm_reasoning(
        self, symbol: str, snap: dict[str, float], signal: Signal
    ) -> str:
        if self.llm_client is None:
            return ""

        pvr = snap["price_vs_resistance"]
        price_desc = (
            "above resistance (confirmed breakout)"       if pvr >= self.ABOVE_RESISTANCE
            else "at resistance (breakout imminent)"      if pvr >= self.AT_RESISTANCE
            else f"approaching resistance ({pvr:.0%} of level)" if pvr >= self.NEAR_RESISTANCE
            else f"below resistance ({pvr:.0%} of level)"
        )

        user_message = (
            f"Stock: {symbol} | Signal: {signal.value}\n"
            f"Volume: {snap['volume_ratio']:.1f}x 20-day average\n"
            f"Price ({snap['price']:.1f}) vs Resistance ({snap['resistance']:.1f}): {price_desc}\n"
            f"ATR (daily volatility): {snap['atr']:.1f}\n\n"
            f"Explain why these breakout indicators support a {signal.value} signal."
        )

        return self.call_llm(self.SYSTEM_PROMPT, user_message, max_tokens=120)
