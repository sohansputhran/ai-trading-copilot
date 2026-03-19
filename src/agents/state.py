"""
Trading System State Schema
============================
This is the SINGLE SOURCE OF TRUTH for data flowing through the multi-agent graph.

Design principle: Every agent reads from state, writes to state.
No agent-to-agent direct calls. All communication goes through state.

Why TypedDict?
- LangGraph requires TypedDict (not dataclasses or Pydantic) for state
- TypedDict gives us type hints without runtime overhead
- All fields must be present in the dict (Optional fields default to None)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, TypedDict

# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────


class Signal(StrEnum):
    """Trading signal. String enum so it serializes cleanly to JSON."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class AgentName(StrEnum):
    """All agents in the system. Single place to manage names."""

    TECHNICAL = "technical_analysis"
    MOMENTUM = "momentum_strategy"
    BREAKOUT = "breakout_strategy"
    AGGREGATOR = "aggregator"


# ─────────────────────────────────────────────
# Agent Output Model
# ─────────────────────────────────────────────


@dataclass
class AgentAnalysis:
    """
    Structured output from a single agent.

    Every agent produces exactly this shape — the aggregator
    knows what to expect regardless of which agent produced it.

    Why dataclass (not TypedDict)?
    - AgentAnalysis is produced BY agents and stored IN state
    - TypedDict is for the state itself (LangGraph requirement)
    - Dataclass is cleaner for objects we create and pass around
    """

    agent_name: str
    signal: Signal
    confidence: float  # 0.0 – 1.0
    reasoning: str  # Human-readable explanation
    key_indicators: dict[str, Any] = field(default_factory=dict)  # Data that drove the decision
    warnings: list[str] = field(default_factory=list)  # Red flags worth surfacing

    def __post_init__(self):
        """Validate confidence is in valid range."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0–1.0, got {self.confidence}")


# ─────────────────────────────────────────────
# Main State Schema
# ─────────────────────────────────────────────


class TradingState(TypedDict):
    """
    State that flows through the entire LangGraph.

    Lifecycle:
    1. Orchestrator populates: symbol, market_data, indicators, scan_timestamp
    2. Agents populate:        technical_analysis, momentum_analysis, breakout_analysis
    3. Aggregator populates:   final_signal, final_confidence, final_reasoning, agent_agreement
    4. Result returned to caller

    Why Optional for agent outputs?
    - Agents run in parallel; we don't know which finishes first
    - Optional signals "not yet populated" vs "ran and returned HOLD"
    - Aggregator checks all three are non-None before running
    """

    # ── Input (set by orchestrator before graph runs) ──────────────────────
    symbol: str
    market_data: dict[str, Any]  # Raw OHLCV + metadata from data pipeline
    indicators: dict[str, Any]  # All computed technical indicators
    scan_timestamp: str  # ISO format string

    # ── Agent outputs (populated during parallel agent execution) ───────────
    technical_analysis: AgentAnalysis | None
    momentum_analysis: AgentAnalysis | None
    breakout_analysis: AgentAnalysis | None

    # ── Final decision (populated by aggregator) ────────────────────────────
    final_signal: Signal | None
    final_confidence: float | None  # Weighted aggregate confidence
    final_reasoning: str | None  # Summary combining all agent reasoning
    agent_agreement: float | None  # 0.0 = all disagree, 1.0 = all agree

    # ── Observability ────────────────────────────────────────────────────────
    errors: list[str]  # Non-fatal errors from any agent


def initial_state(symbol: str, market_data: dict, indicators: dict) -> TradingState:
    """
    Factory function that creates a clean initial state.

    Always use this instead of constructing TradingState manually.
    Ensures all Optional fields start as None and errors list is empty.
    """
    return TradingState(
        symbol=symbol,
        market_data=market_data,
        indicators=indicators,
        scan_timestamp=datetime.utcnow().isoformat(),
        technical_analysis=None,
        momentum_analysis=None,
        breakout_analysis=None,
        final_signal=None,
        final_confidence=None,
        final_reasoning=None,
        agent_agreement=None,
        errors=[],
    )
