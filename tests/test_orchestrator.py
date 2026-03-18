"""
Tests: Full Orchestrator Pipeline (requires LangGraph)
========================================================
Tests the complete LangGraph StateGraph end-to-end using mock agents.

Dependencies: pip install langgraph structlog pytest

Run with:
    pytest tests/test_orchestrator.py -v

Why separate from test_agents.py?
- test_agents.py has zero external deps — runs anywhere instantly
- This file requires langgraph — skip it if langgraph isn't installed yet
  (pytest will skip automatically with: pytest -m "not integration")
"""

import pytest

# Skip entire file gracefully if langgraph not installed
langgraph = pytest.importorskip("langgraph", reason="langgraph not installed")

from src.agents.orchestrator import MultiAgentOrchestrator
from src.agents.state import AgentAnalysis, Signal, TradingState

# ─────────────────────────────────────────────
# Mock agents
# ─────────────────────────────────────────────

class MockAgent:
    def __init__(self, name: str, signal: Signal, confidence: float):
        self.agent_name  = name
        self._signal     = signal
        self._confidence = confidence

    def analyze(self, state: TradingState) -> AgentAnalysis:
        return AgentAnalysis(
            agent_name=self.agent_name, signal=self._signal,
            confidence=self._confidence, reasoning=f"Mock: {self.agent_name}",
        )

class FailingAgent:
    agent_name = "failing_agent"
    def analyze(self, state):
        raise RuntimeError("Simulated failure")


# ─────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def indicators():
    return {
        "price": 2410.0, "rsi": 25.0, "macd": -1.5,
        "macd_signal": -2.2, "bb_position": 0.08, "volume_ratio": 1.3,
    }

@pytest.fixture
def market_data():
    return {"close": 2410.0, "volume": 2_000_000}


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

def test_unanimous_buy_flows_through_graph(market_data, indicators):
    orch = MultiAgentOrchestrator(
        MockAgent("technical_analysis", Signal.BUY, 0.80),
        MockAgent("momentum_strategy",  Signal.BUY, 0.75),
        MockAgent("breakout_strategy",  Signal.BUY, 0.70),
    )
    result = orch.analyze("RELIANCE.NS", market_data, indicators)

    assert result["final_signal"]     == Signal.BUY
    assert result["agent_agreement"]  == 1.0
    assert result["final_confidence"]  > 0.6
    assert result["technical_analysis"] is not None
    assert result["momentum_analysis"]  is not None
    assert result["breakout_analysis"]  is not None


def test_all_disagree_gives_hold(market_data, indicators):
    orch = MultiAgentOrchestrator(
        MockAgent("technical_analysis", Signal.BUY,  0.70),
        MockAgent("momentum_strategy",  Signal.SELL, 0.70),
        MockAgent("breakout_strategy",  Signal.HOLD, 0.60),
    )
    result = orch.analyze("RELIANCE.NS", market_data, indicators)
    assert result["final_signal"] == Signal.HOLD


def test_failing_agent_does_not_crash_pipeline(market_data, indicators):
    orch = MultiAgentOrchestrator(
        FailingAgent(),
        MockAgent("momentum_strategy", Signal.BUY, 0.75),
        MockAgent("breakout_strategy", Signal.BUY, 0.70),
    )
    result = orch.analyze("RELIANCE.NS", market_data, indicators)

    assert len(result["errors"]) > 0
    assert result["final_signal"] is not None


def test_all_required_state_fields_present(market_data, indicators):
    orch = MultiAgentOrchestrator(
        MockAgent("technical_analysis", Signal.BUY, 0.75),
        MockAgent("momentum_strategy",  Signal.BUY, 0.75),
        MockAgent("breakout_strategy",  Signal.BUY, 0.75),
    )
    result = orch.analyze("RELIANCE.NS", market_data, indicators)

    for field in [
        "symbol", "final_signal", "final_confidence",
        "final_reasoning", "agent_agreement", "errors",
        "technical_analysis", "momentum_analysis", "breakout_analysis",
    ]:
        assert field in result, f"Missing: {field}"
