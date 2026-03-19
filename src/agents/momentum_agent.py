"""
Momentum Strategy Agent
========================
Specializes in trend-following signals using EMA crossovers and ADX strength.

This agent answers one question:
    "Is there a strong, established trend worth following right now?"

WHY MOMENTUM AS A SEPARATE AGENT?
Momentum trading is a fundamentally different philosophy from oscillator analysis
(Technical Agent). Oscillators ask "is the stock stretched?" — they work best in
sideways/ranging markets. Momentum asks "is there a strong trend to ride?" —
it works best when markets are trending.

A stock can be RSI-overbought (Technical Agent says SELL) AND have a strong upward
momentum (Momentum Agent says BUY). This disagreement is valuable information —
the aggregator will see low agreement and reduce confidence, avoiding a bad trade.
That's exactly why we use multiple specialized agents.

Indicators used (all from get_latest_signals()):
    'ema_20'       → 20-day EMA (fast, short-term trend)
    'ema_50'       → 50-day EMA (slow, medium-term trend)
    'ema_cross'    → +1 if EMA20 > EMA50, -1 if below
    'adx'          → ADX value (trend strength, direction-independent)
    'rsi'          → Used as a filter only (avoid chasing extreme moves)
    'price'        → Current price vs EMAs

Scoring logic (max 5.0 pts):
    EMA crossover direction:     2.0 pts  (primary signal)
    ADX trend strength:          2.0 pts  (conviction filter)
    Price position vs EMA20:     1.0 pts  (momentum confirmation)
"""

from typing import Any

import structlog

from src.agents.base_agent import BaseStrategyAgent
from src.agents.state import AgentAnalysis, AgentName, Signal, TradingState

logger = structlog.get_logger()


class MomentumStrategyAgent(BaseStrategyAgent):
    """
    Trend-following agent using EMA crossovers and ADX strength.
    Combines rule-based signals with optional LLM reasoning enrichment.
    """

    agent_name = AgentName.MOMENTUM.value

    # ── Thresholds ────────────────────────────────────────────────────────────
    ADX_WEAK_TREND = 20.0  # Below this → no clear trend, avoid momentum trades
    ADX_STRONG_TREND = 40.0  # Above this → strong trend, high conviction

    RSI_CHASE_LIMIT = 75.0  # Don't chase a BUY when RSI already this high
    RSI_CRASH_LIMIT = 25.0  # Don't chase a SELL when RSI already this low

    SYSTEM_PROMPT = (
        "You are a momentum trader specializing in trend-following strategies. "
        "Explain in 2 sentences why the EMA crossover and ADX values support the stated signal. "
        "Be specific: mention the actual ADX value and what it tells us about trend strength. "
        "Do NOT recommend actual trades. Do NOT repeat the signal word at the start."
    )

    def analyze(self, state: TradingState) -> AgentAnalysis:
        indicators = state["indicators"]
        symbol = state["symbol"]
        self.log = logger.bind(agent=self.agent_name, symbol=symbol)

        snapshot = self._extract_indicators(indicators)
        signal, confidence, rule_text, warnings = self._compute_signal(snapshot)
        reasoning = self._get_llm_reasoning(symbol, snapshot, signal) or rule_text

        return AgentAnalysis(
            agent_name=self.agent_name,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            key_indicators=snapshot,
            warnings=warnings,
        )

    # ─────────────────────────────────────────────
    # Private: indicator extraction
    # ─────────────────────────────────────────────

    def _extract_indicators(self, indicators: dict[str, Any]) -> dict[str, float]:
        """Pull only the indicators this agent uses."""
        return {
            "ema_20": self.safe_get(indicators, "ema_20", 0.0),
            "ema_50": self.safe_get(indicators, "ema_50", 0.0),
            "ema_cross": self.safe_get(indicators, "ema_cross", 0.0),
            "adx": self.safe_get(indicators, "adx", 0.0),
            "rsi": self.safe_get(indicators, "rsi", 50.0),
            "price": self.safe_get(indicators, "price", 0.0),
        }

    # ─────────────────────────────────────────────
    # Private: rule-based signal logic
    # ─────────────────────────────────────────────

    def _compute_signal(self, snap: dict[str, float]) -> tuple[Signal, float, str, list[str]]:
        """
        Multi-factor momentum scoring.

        Key insight: ADX is the gate. If ADX < 20, there is no trend to follow
        regardless of what the EMA crossover says. We return HOLD immediately.
        This prevents momentum agents from generating false signals in choppy markets.

        Scoring:
            EMA crossover:        2.0 pts  — direction of the trend
            ADX strength:         2.0 pts  — how strong is that trend
            Price vs EMA20:       1.0 pts  — is price leading or lagging?
        """
        ema_cross = snap["ema_cross"]
        adx = snap["adx"]
        rsi = snap["rsi"]
        price = snap["price"]
        ema_20 = snap["ema_20"]

        bull_score = 0.0
        bear_score = 0.0
        reasons: list[str] = []
        warnings: list[str] = []

        # ── Gate: ADX must show a trend exists ───────────────────────────────
        if adx < self.ADX_WEAK_TREND:
            self.log.debug("adx_gate_failed", adx=adx)
            return (
                Signal.HOLD,
                0.2,
                f"No clear trend — ADX={adx:.1f} is below {self.ADX_WEAK_TREND} "
                f"(choppy/sideways market). Momentum strategy not applicable.",
                [f"ADX={adx:.1f} indicates no trending conditions — avoid momentum trades"],
            )

        # ── EMA crossover direction (2.0 pts) ─────────────────────────────────
        if ema_cross > 0:
            bull_score += 2.0
            reasons.append(
                f"EMA20 ({snap['ema_20']:.1f}) above EMA50 ({snap['ema_50']:.1f}) — bullish cross"
            )
        elif ema_cross < 0:
            bear_score += 2.0
            reasons.append(
                f"EMA20 ({snap['ema_20']:.1f}) below EMA50 ({snap['ema_50']:.1f}) — bearish cross"
            )

        # ── ADX strength (2.0 pts) ────────────────────────────────────────────
        if adx >= self.ADX_STRONG_TREND:
            # Strong trend — full points
            adx_pts = 2.0
            reasons.append(f"ADX={adx:.1f} (strong trend ≥{self.ADX_STRONG_TREND})")
        else:
            # Developing trend — partial points proportional to ADX
            # Maps ADX range [20, 40] → [0.5, 2.0] pts
            adx_pts = 0.5 + 1.5 * (adx - self.ADX_WEAK_TREND) / (
                self.ADX_STRONG_TREND - self.ADX_WEAK_TREND
            )
            reasons.append(f"ADX={adx:.1f} (developing trend)")

        if ema_cross > 0:
            bull_score += adx_pts
        elif ema_cross < 0:
            bear_score += adx_pts

        # ── Price vs EMA20 (1.0 pts) ──────────────────────────────────────────
        # Price above EMA20 confirms bullish momentum; below confirms bearish
        if ema_20 > 0:
            if price > ema_20 and ema_cross > 0:
                bull_score += 1.0
                reasons.append(f"Price ({price:.1f}) above EMA20 — momentum confirmed")
            elif price < ema_20 and ema_cross < 0:
                bear_score += 1.0
                reasons.append(f"Price ({price:.1f}) below EMA20 — downtrend confirmed")

        # ── RSI chase filter ──────────────────────────────────────────────────
        # Don't chase extremely stretched moves — high risk of reversal
        if bull_score > bear_score and rsi > self.RSI_CHASE_LIMIT:
            warnings.append(
                f"RSI={rsi:.1f} is very high — trend may be overextended, "
                f"reversal risk elevated"
            )
        if bear_score > bull_score and rsi < self.RSI_CRASH_LIMIT:
            warnings.append(
                f"RSI={rsi:.1f} is very low — downtrend may be overextended, "
                f"bounce risk elevated"
            )

        # ── Determine signal + confidence ─────────────────────────────────────
        max_possible = 5.0

        if bull_score > bear_score:
            signal = Signal.BUY
            confidence = round(min(bull_score / max_possible, 1.0), 3)
            rule_text = (
                f"Bullish momentum signal. {'; '.join(reasons)}. "
                f"Score: {bull_score:.1f}/{max_possible}"
            )
        elif bear_score > bull_score:
            signal = Signal.SELL
            confidence = round(min(bear_score / max_possible, 1.0), 3)
            rule_text = (
                f"Bearish momentum signal. {'; '.join(reasons)}. "
                f"Score: {bear_score:.1f}/{max_possible}"
            )
        else:
            signal = Signal.HOLD
            confidence = 0.2
            rule_text = "No clear momentum direction — EMA signals are conflicting."

        self.log.debug(
            "rule_signal_computed",
            signal=signal,
            bull_score=bull_score,
            bear_score=bear_score,
            confidence=confidence,
            adx=adx,
        )

        return signal, confidence, rule_text, warnings

    # ─────────────────────────────────────────────
    # Private: LLM reasoning enrichment
    # ─────────────────────────────────────────────

    def _get_llm_reasoning(self, symbol: str, snap: dict[str, float], signal: Signal) -> str:
        if self.llm_client is None:
            return ""

        cross_desc = (
            "bullish (EMA20 above EMA50)"
            if snap["ema_cross"] > 0
            else "bearish (EMA20 below EMA50)" if snap["ema_cross"] < 0 else "neutral (EMAs equal)"
        )
        trend_desc = (
            "strong"
            if snap["adx"] >= self.ADX_STRONG_TREND
            else "developing" if snap["adx"] >= self.ADX_WEAK_TREND else "weak/absent"
        )

        user_message = (
            f"Stock: {symbol} | Signal: {signal.value}\n"
            f"EMA20: {snap['ema_20']:.1f} vs EMA50: {snap['ema_50']:.1f} → {cross_desc}\n"
            f"ADX: {snap['adx']:.1f} ({trend_desc} trend)\n"
            f"Price: {snap['price']:.1f} vs EMA20: {snap['ema_20']:.1f}\n\n"
            f"Explain why these momentum indicators support a {signal.value} signal."
        )

        return self.call_llm(self.SYSTEM_PROMPT, user_message, max_tokens=120)
