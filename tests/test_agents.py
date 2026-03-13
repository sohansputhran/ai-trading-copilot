"""
Tests: Agent Logic Only (No LangGraph required)
================================================
These tests validate agent signal logic and state schema independently.
No LangGraph, no HuggingFace, no network calls needed.

Dependencies: pytest, structlog (both lightweight, pip install them)

Run with:
    pytest tests/test_agents.py -v

What's tested:
- TradingState schema and initial_state() factory
- AgentAnalysis dataclass validation
- TechnicalAnalysisAgent rule-based signal logic
- Aggregator math (weighted confidence, agreement scoring)
- BaseStrategyAgent safe_get() with missing keys
"""

import pytest
from src.agents.state import (
    TradingState, AgentAnalysis, Signal, AgentName, initial_state
)
from src.agents.technical_agent import TechnicalAnalysisAgent
from src.agents.aggregator import aggregate


# ─────────────────────────────────────────────
# Fixtures — use exact keys from get_latest_signals()
# ─────────────────────────────────────────────

@pytest.fixture
def oversold_indicators():
    return {
        "price":        2410.0,
        "rsi":            25.0,   # Oversold → BUY signal
        "macd":           -1.5,
        "macd_signal":    -2.2,   # macd > macd_signal → positive spread → bullish
        "bb_position":     0.08,  # Near lower band
        "volume_ratio":    1.3,
    }

@pytest.fixture
def overbought_indicators():
    return {
        "price":        2598.0,
        "rsi":            78.0,   # Overbought → SELL signal
        "macd":            3.5,
        "macd_signal":     4.1,   # macd < macd_signal → negative spread → bearish
        "bb_position":     0.92,  # Near upper band
        "volume_ratio":    0.9,
    }

@pytest.fixture
def neutral_indicators():
    return {
        "price":        2500.0,
        "rsi":            52.0,   # Mild bull but no strong signal
        "macd":            0.3,
        "macd_signal":     0.4,   # Near-zero spread
        "bb_position":     0.5,   # Mid-band
        "volume_ratio":    1.1,
    }


# ─────────────────────────────────────────────
# State schema tests
# ─────────────────────────────────────────────

def test_initial_state_has_all_keys(oversold_indicators):
    """initial_state() must populate all required TradingState fields."""
    state = initial_state("RELIANCE.NS", {"close": 2410.0}, oversold_indicators)

    assert state["symbol"]              == "RELIANCE.NS"
    assert state["indicators"]          == oversold_indicators
    assert state["technical_analysis"]  is None   # Not yet populated
    assert state["momentum_analysis"]   is None
    assert state["breakout_analysis"]   is None
    assert state["final_signal"]        is None
    assert state["errors"]              == []


def test_agent_analysis_confidence_validation():
    """AgentAnalysis should reject confidence outside 0.0–1.0."""
    with pytest.raises(ValueError):
        AgentAnalysis(
            agent_name="test", signal=Signal.BUY,
            confidence=1.5,      # Invalid — over 1.0
            reasoning="test",
        )


def test_agent_analysis_valid():
    """Valid AgentAnalysis should construct without error."""
    analysis = AgentAnalysis(
        agent_name    = "technical_analysis",
        signal        = Signal.BUY,
        confidence    = 0.75,
        reasoning     = "RSI oversold",
        key_indicators= {"rsi": 25.0},
        warnings      = [],
    )
    assert analysis.signal     == Signal.BUY
    assert analysis.confidence == 0.75


# ─────────────────────────────────────────────
# TechnicalAnalysisAgent tests
# ─────────────────────────────────────────────

def test_technical_buy_on_oversold(oversold_indicators):
    """RSI=25 + near lower BB + positive MACD spread → BUY."""
    agent  = TechnicalAnalysisAgent(llm_client=None)
    state  = initial_state("RELIANCE.NS", {}, oversold_indicators)
    result = agent.analyze(state)

    assert result.signal     == Signal.BUY
    assert result.confidence  > 0.5
    assert "RSI" in result.reasoning
    assert result.agent_name == AgentName.TECHNICAL.value


def test_technical_sell_on_overbought(overbought_indicators):
    """RSI=78 + near upper BB + negative MACD spread → SELL."""
    agent  = TechnicalAnalysisAgent(llm_client=None)
    state  = initial_state("RELIANCE.NS", {}, overbought_indicators)
    result = agent.analyze(state)

    assert result.signal     == Signal.SELL
    assert result.confidence  > 0.5


def test_technical_low_confidence_on_neutral(neutral_indicators):
    """RSI=52 scores a weak bull bias (0.5/4.0 = 0.125 confidence).
    Signal may be BUY or HOLD but confidence must be very low — not tradeable."""
    agent  = TechnicalAnalysisAgent(llm_client=None)
    state  = initial_state("RELIANCE.NS", {}, neutral_indicators)
    result = agent.analyze(state)

    # Weak RSI bias alone is not enough to trade — confidence should be very low
    assert result.confidence < 0.3
    # Aggregator will reject this anyway (below MIN_CONFIDENCE_TO_TRADE=0.60)


def test_technical_handles_missing_indicators():
    """safe_get() should use defaults for missing keys — no KeyError."""
    agent = TechnicalAnalysisAgent(llm_client=None)
    # Deliberately pass only one key — everything else defaults
    state = initial_state("TEST.NS", {}, {"rsi": 28.0})

    result = agent.analyze(state)   # Must not raise
    assert result.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)


def test_technical_key_indicators_populated(oversold_indicators):
    """key_indicators on AgentAnalysis should contain the snapshot used."""
    agent  = TechnicalAnalysisAgent(llm_client=None)
    state  = initial_state("RELIANCE.NS", {}, oversold_indicators)
    result = agent.analyze(state)

    assert "rsi"         in result.key_indicators
    assert "bb_position" in result.key_indicators
    assert result.key_indicators["rsi"] == 25.0


def test_technical_rsi_warning_near_oversold():
    """RSI between 30–35 should generate a warning, not a BUY signal."""
    agent  = TechnicalAnalysisAgent(llm_client=None)
    state  = initial_state("TEST.NS", {}, {
        "rsi": 32.0,           # Near but not yet oversold
        "macd": 0.1,
        "macd_signal": 0.0,
        "bb_position": 0.3,
        "price": 1000.0,
    })
    result = agent.analyze(state)

    assert any("approaching oversold" in w for w in result.warnings)


# ─────────────────────────────────────────────
# Aggregator tests (pure math — no agents needed)
# ─────────────────────────────────────────────

def _make_state_with_analyses(
    tech: tuple, mom: tuple, brk: tuple
) -> TradingState:
    """Helper: build a state with three AgentAnalysis objects already populated."""
    state = initial_state("TEST.NS", {}, {})
    state["technical_analysis"] = AgentAnalysis(
        agent_name="technical_analysis", signal=tech[0], confidence=tech[1], reasoning=""
    )
    state["momentum_analysis"] = AgentAnalysis(
        agent_name="momentum_strategy", signal=mom[0], confidence=mom[1], reasoning=""
    )
    state["breakout_analysis"] = AgentAnalysis(
        agent_name="breakout_strategy", signal=brk[0], confidence=brk[1], reasoning=""
    )
    return state


def test_aggregator_unanimous_buy():
    """All three BUY → final BUY with agreement=1.0."""
    state   = _make_state_with_analyses(
        (Signal.BUY, 0.80), (Signal.BUY, 0.75), (Signal.BUY, 0.70)
    )
    updates = aggregate(state)

    assert updates["final_signal"]    == Signal.BUY
    assert updates["agent_agreement"] == 1.0
    assert updates["final_confidence"] > 0.6


def test_aggregator_all_disagree_gives_hold():
    """BUY + SELL + HOLD → no majority → HOLD."""
    state   = _make_state_with_analyses(
        (Signal.BUY, 0.70), (Signal.SELL, 0.70), (Signal.HOLD, 0.60)
    )
    updates = aggregate(state)

    assert updates["final_signal"] == Signal.HOLD


def test_aggregator_low_confidence_gives_hold():
    """Even unanimous BUY with low confidence should → HOLD (below threshold)."""
    state   = _make_state_with_analyses(
        (Signal.BUY, 0.30), (Signal.BUY, 0.25), (Signal.BUY, 0.20)
    )
    updates = aggregate(state)

    # Weighted confidence will be ~0.25 — below MIN_CONFIDENCE_TO_TRADE (0.60)
    assert updates["final_signal"]     == Signal.HOLD
    assert updates["final_confidence"]  < 0.60


def test_aggregator_reasoning_contains_all_agents():
    """final_reasoning should mention all three agent names (uppercased in output)."""
    state   = _make_state_with_analyses(
        (Signal.BUY, 0.80), (Signal.BUY, 0.75), (Signal.BUY, 0.70)
    )
    updates = aggregate(state)

    reasoning = updates["final_reasoning"].upper()
    assert "TECHNICAL_ANALYSIS" in reasoning
    assert "MOMENTUM_STRATEGY"  in reasoning
    assert "BREAKOUT_STRATEGY"  in reasoning


def test_aggregator_missing_all_analyses():
    """If all agents failed and state has no analyses, should HOLD gracefully."""
    state   = initial_state("TEST.NS", {}, {})
    # technical_analysis, momentum_analysis, breakout_analysis all None
    updates = aggregate(state)

    assert updates["final_signal"]     == Signal.HOLD
    assert updates["final_confidence"] == 0.0
    assert len(updates["errors"])       > 0