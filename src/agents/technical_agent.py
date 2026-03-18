"""
Technical Analysis Agent
=========================
Specializes in oscillator-based signals: RSI, MACD, Bollinger Bands.

This agent answers one question:
    "Based on technical oscillators, is this stock overbought, oversold, or neutral?"

Indicator key mapping (matches indicators.py → get_latest_signals() output exactly):
    'rsi'          → latest RSI value
    'macd'         → MACD line value
    'macd_signal'  → Signal line value
    'bb_position'  → 0.0–1.0 float: 0 = at lower band, 1 = at upper band
    'volume_ratio' → current volume / 20-day avg volume

Note: indicators.py does NOT expose MACD_Histogram or raw BB_Upper/BB_Lower
in get_latest_signals(). We derive what we need from the available keys.

This agent deliberately does NOT look at raw price levels or volume breakouts
(that's Breakout Agent's job). Specialization makes reasoning cleaner.
"""

from typing import Any

import structlog

from src.agents.base_agent import BaseStrategyAgent
from src.agents.state import AgentAnalysis, AgentName, Signal, TradingState

logger = structlog.get_logger()


class TechnicalAnalysisAgent(BaseStrategyAgent):
    """
    Oscillator-focused technical analysis agent.
    Combines rule-based signals with optional LLM reasoning enrichment.
    """

    agent_name = AgentName.TECHNICAL.value

    # ── Signal thresholds (standard TA practice) ──────────────────────────────
    RSI_OVERSOLD    = 30.0
    RSI_OVERBOUGHT  = 70.0
    RSI_MILD_BULL   = 50.0   # Above midpoint = mild bullish bias

    # bb_position thresholds (0.0 = at lower band, 1.0 = at upper band)
    BB_NEAR_LOWER   = 0.15   # Price near lower band → oversold territory
    BB_NEAR_UPPER   = 0.85   # Price near upper band → overbought territory

    # System prompt for LLM reasoning enrichment
    SYSTEM_PROMPT = (
        "You are a technical analyst specializing in momentum oscillators. "
        "Explain in 2 sentences why the given indicators support the stated signal. "
        "Be specific: mention the actual RSI value and what it means. "
        "Do NOT recommend actual trades. Do NOT repeat the signal word at the start."
    )

    def analyze(self, state: TradingState) -> AgentAnalysis:
        """
        Entry point called by the LangGraph node.

        Strategy:
        1. Extract the indicators this agent uses from state
        2. Compute rule-based signal + confidence
        3. Optionally enrich with LLM natural language reasoning
        4. Return typed AgentAnalysis
        """
        indicators = state["indicators"]
        symbol     = state["symbol"]
        self.log   = logger.bind(agent=self.agent_name, symbol=symbol)

        snapshot                     = self._extract_indicators(indicators)
        signal, confidence, rule_text, warnings = self._compute_signal(snapshot)
        reasoning                    = self._get_llm_reasoning(symbol, snapshot, signal) or rule_text

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
        """
        Pull only the indicators this agent uses.

        Key names match get_latest_signals() output from indicators.py:
            'rsi', 'macd', 'macd_signal', 'bb_position', 'volume_ratio', 'price'
        """
        return {
            "rsi":          self.safe_get(indicators, "rsi",          50.0),
            "macd":         self.safe_get(indicators, "macd",          0.0),
            "macd_signal":  self.safe_get(indicators, "macd_signal",   0.0),
            "bb_position":  self.safe_get(indicators, "bb_position",   0.5),
            "price":        self.safe_get(indicators, "price",         0.0),
        }

    # ─────────────────────────────────────────────
    # Private: rule-based signal logic
    # ─────────────────────────────────────────────

    def _compute_signal(
        self, snap: dict[str, float]
    ) -> tuple[Signal, float, str, list[str]]:
        """
        Multi-factor scoring system.

        Each condition contributes to bull_score or bear_score.
        Confidence is proportional to how many conditions fired.
        This avoids the "one indicator = one trade" trap.

        Scoring (max 4.0 pts each side):
            RSI oversold/overbought:    2.0 pts  (primary signal)
            MACD line vs signal:        1.5 pts  (momentum direction)
            Bollinger Band position:    0.5 pts  (volatility context)
        """
        rsi         = snap["rsi"]
        macd        = snap["macd"]
        macd_signal = snap["macd_signal"]
        bb_pos      = snap["bb_position"]

        # Derived: MACD histogram equivalent (line minus signal)
        macd_hist_equiv = macd - macd_signal

        bull_score = 0.0
        bear_score = 0.0
        reasons:  list[str] = []
        warnings: list[str] = []

        # ── RSI (2.0 pts) ─────────────────────────────────────────────────────
        if rsi < self.RSI_OVERSOLD:
            bull_score += 2.0
            reasons.append(f"RSI={rsi:.1f} (oversold, below {self.RSI_OVERSOLD})")
        elif rsi > self.RSI_OVERBOUGHT:
            bear_score += 2.0
            reasons.append(f"RSI={rsi:.1f} (overbought, above {self.RSI_OVERBOUGHT})")
        elif rsi > self.RSI_MILD_BULL:
            bull_score += 0.5
            reasons.append(f"RSI={rsi:.1f} (above midpoint, mild bull bias)")

        # ── MACD line vs signal (1.5 pts) ─────────────────────────────────────
        if macd_hist_equiv > 0.2:
            bull_score += 1.5
            reasons.append(f"MACD ({macd:.2f}) above signal ({macd_signal:.2f}) — bullish")
        elif macd_hist_equiv < -0.2:
            bear_score += 1.5
            reasons.append(f"MACD ({macd:.2f}) below signal ({macd_signal:.2f}) — bearish")

        # ── Bollinger Band position (0.5 pts) ─────────────────────────────────
        if bb_pos <= self.BB_NEAR_LOWER:
            bull_score += 0.5
            reasons.append(f"Price near lower Bollinger Band (bb_pos={bb_pos:.2f})")
        elif bb_pos >= self.BB_NEAR_UPPER:
            bear_score += 0.5
            reasons.append(f"Price near upper Bollinger Band (bb_pos={bb_pos:.2f})")

        # ── Warnings (non-fatal, surface to aggregator) ───────────────────────
        if 30 < rsi < 35:
            warnings.append(f"RSI={rsi:.1f} approaching oversold zone — watch closely")
        if 65 < rsi < 70:
            warnings.append(f"RSI={rsi:.1f} approaching overbought zone — watch closely")

        # ── Determine signal + confidence ─────────────────────────────────────
        max_possible = 4.0

        if bull_score > bear_score:
            signal     = Signal.BUY
            confidence = round(min(bull_score / max_possible, 1.0), 3)
            rule_text  = (
                f"Bullish oscillator signal. {'; '.join(reasons)}. "
                f"Score: {bull_score:.1f}/{max_possible}"
            )
        elif bear_score > bull_score:
            signal     = Signal.SELL
            confidence = round(min(bear_score / max_possible, 1.0), 3)
            rule_text  = (
                f"Bearish oscillator signal. {'; '.join(reasons)}. "
                f"Score: {bear_score:.1f}/{max_possible}"
            )
        else:
            signal     = Signal.HOLD
            confidence = 0.3
            rule_text  = "Neutral oscillator conditions — no clear signal."

        self.log.debug(
            "rule_signal_computed",
            signal=signal, bull_score=bull_score,
            bear_score=bear_score, confidence=confidence,
        )

        return signal, confidence, rule_text, warnings

    # ─────────────────────────────────────────────
    # Private: LLM reasoning enrichment
    # ─────────────────────────────────────────────

    def _get_llm_reasoning(
        self, symbol: str, snap: dict[str, float], signal: Signal
    ) -> str:
        """
        Ask the LLM to explain the pre-computed signal in plain English.

        We pass the signal to the LLM — it explains, not decides.
        This is "LLM for language, rules for logic."
        """
        if self.llm_client is None:
            return ""

        bb_desc = (
            "near lower band (oversold zone)" if snap["bb_pos"] <= self.BB_NEAR_LOWER
            else "near upper band (overbought zone)" if snap["bb_pos"] >= self.BB_NEAR_UPPER
            else "within bands (neutral)"
        ) if "bb_pos" in snap else (
            "near lower band" if snap["bb_position"] <= self.BB_NEAR_LOWER
            else "near upper band" if snap["bb_position"] >= self.BB_NEAR_UPPER
            else "within bands"
        )

        user_message = (
            f"Stock: {symbol} | Signal: {signal.value}\n"
            f"RSI: {snap['rsi']:.1f}\n"
            f"MACD: {snap['macd']:.2f} vs Signal: {snap['macd_signal']:.2f}\n"
            f"Bollinger Band position: {bb_desc}\n\n"
            f"Explain why these oscillators support a {signal.value} signal."
        )

        return self.call_llm(self.SYSTEM_PROMPT, user_message, max_tokens=120)
