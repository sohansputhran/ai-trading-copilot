"""
Multi-Agent Orchestrator
=========================
LangGraph StateGraph that coordinates all strategy agents.

Graph topology:
    START
      │
      ├──► technical_agent ──┐
      ├──► momentum_agent  ──┼──► aggregator ──► END
      └──► breakout_agent  ──┘

The fan-out (parallel execution) is handled by LangGraph's Send API.
All three agents run concurrently; aggregator waits for all three.

How LangGraph executes this:
1. START node runs, sends the state to all three agent nodes simultaneously
2. Each agent node receives a copy of state, writes its analysis back
3. LangGraph merges the three updated states (each wrote to different keys, no conflicts)
4. Aggregator node runs with the fully-populated state
5. Aggregator writes final decision fields
6. Graph returns final state to caller

Why this topology over alternatives:
- Sequential (A → B → C): ~3x slower, agents influence each other's reasoning
- Supervisor (one LLM decides who runs): adds latency and LLM cost for no benefit
- Parallel fan-out: fastest, independent opinions, clean state merging
"""

from typing import Any

import structlog
from langgraph.graph import END, START, StateGraph

from src.agents.aggregator import aggregate
from src.agents.state import TradingState, initial_state

logger = structlog.get_logger()


def build_orchestrator(
    technical_agent,
    momentum_agent,
    breakout_agent,
) -> Any:
    """
    Compile and return the LangGraph StateGraph.

    Why a factory function (not a class)?
    - Agents are injected (dependency injection) — easy to swap in tests
    - The graph is compiled once and reused across multiple stock analyses
    - Compiled graph is immutable; you don't need to rebuild it for each stock

    Args:
        technical_agent:  Instance of TechnicalAnalysisAgent
        momentum_agent:   Instance of MomentumStrategyAgent
        breakout_agent:   Instance of BreakoutStrategyAgent

    Returns:
        Compiled LangGraph application (callable)
    """

    # ── Define node wrapper functions ─────────────────────────────────────────
    # LangGraph node functions receive state and return a dict of UPDATES.
    # LangGraph merges the updates into state — you don't return the full state.
    # This is why parallel execution works: each node writes to different keys.

    def run_technical(state: TradingState) -> dict:
        """Technical Analysis Agent node."""
        log = logger.bind(symbol=state["symbol"], node="technical_agent")
        log.info("node_start")
        try:
            analysis = technical_agent.analyze(state)
            log.info("node_complete", signal=analysis.signal, confidence=analysis.confidence)
            return {"technical_analysis": analysis}
        except Exception as e:
            log.error("node_failed", error=str(e))
            return {"errors": state["errors"] + [f"technical_agent: {str(e)}"]}

    def run_momentum(state: TradingState) -> dict:
        """Momentum Strategy Agent node."""
        log = logger.bind(symbol=state["symbol"], node="momentum_agent")
        log.info("node_start")
        try:
            analysis = momentum_agent.analyze(state)
            log.info("node_complete", signal=analysis.signal, confidence=analysis.confidence)
            return {"momentum_analysis": analysis}
        except Exception as e:
            log.error("node_failed", error=str(e))
            return {"errors": state["errors"] + [f"momentum_agent: {str(e)}"]}

    def run_breakout(state: TradingState) -> dict:
        """Breakout Strategy Agent node."""
        log = logger.bind(symbol=state["symbol"], node="breakout_agent")
        log.info("node_start")
        try:
            analysis = breakout_agent.analyze(state)
            log.info("node_complete", signal=analysis.signal, confidence=analysis.confidence)
            return {"breakout_analysis": analysis}
        except Exception as e:
            log.error("node_failed", error=str(e))
            return {"errors": state["errors"] + [f"breakout_agent: {str(e)}"]}

    def run_aggregator(state: TradingState) -> dict:
        """Aggregator node — combines all agent outputs."""
        log = logger.bind(symbol=state["symbol"], node="aggregator")
        log.info("node_start")
        try:
            updates = aggregate(state)
            log.info("node_complete", final_signal=updates.get("final_signal"))
            return updates
        except Exception as e:
            log.error("node_failed", error=str(e))
            return {
                "final_signal":     "HOLD",
                "final_confidence": 0.0,
                "final_reasoning":  f"Aggregator failed: {str(e)}. Defaulting to HOLD.",
                "agent_agreement":  0.0,
                "errors": state["errors"] + [f"aggregator: {str(e)}"],
            }

    # ── Build the graph ───────────────────────────────────────────────────────

    graph = StateGraph(TradingState)

    # Add all nodes
    graph.add_node("technical_agent", run_technical)
    graph.add_node("momentum_agent",  run_momentum)
    graph.add_node("breakout_agent",  run_breakout)
    graph.add_node("aggregator",      run_aggregator)

    # Fan-out: START → all three agents in parallel
    graph.add_edge(START, "technical_agent")
    graph.add_edge(START, "momentum_agent")
    graph.add_edge(START, "breakout_agent")

    # Fan-in: all three agents → aggregator
    # LangGraph automatically waits for all incoming edges before running a node
    graph.add_edge("technical_agent", "aggregator")
    graph.add_edge("momentum_agent",  "aggregator")
    graph.add_edge("breakout_agent",  "aggregator")

    # Aggregator is the terminal node
    graph.add_edge("aggregator", END)

    return graph.compile()


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

class MultiAgentOrchestrator:
    """
    High-level interface for running multi-agent analysis.

    This class wraps the compiled LangGraph and provides a clean
    interface for the Streamlit dashboard and any other callers.

    Usage:
        orchestrator = MultiAgentOrchestrator(technical, momentum, breakout)
        result = orchestrator.analyze("RELIANCE.NS", market_data, indicators)
        print(result.final_signal)  # Signal.BUY
    """

    def __init__(self, technical_agent, momentum_agent, breakout_agent):
        self.graph = build_orchestrator(technical_agent, momentum_agent, breakout_agent)
        self.log   = logger.bind(component="orchestrator")

    def analyze(
        self,
        symbol: str,
        market_data: dict,
        indicators: dict,
    ) -> TradingState:
        """
        Run the full multi-agent analysis pipeline for one stock.

        Args:
            symbol:      Stock symbol e.g. "RELIANCE.NS"
            market_data: OHLCV dict from data pipeline
            indicators:  Computed technical indicators dict

        Returns:
            Final TradingState with all agent analyses and final decision populated.
        """
        self.log.info("analysis_start", symbol=symbol)

        state = initial_state(symbol, market_data, indicators)

        # invoke() is synchronous — runs the graph and blocks until complete
        # For async use, call ainvoke() instead
        final_state = self.graph.invoke(state)

        self.log.info(
            "analysis_complete",
            symbol=symbol,
            final_signal=final_state.get("final_signal"),
            final_confidence=final_state.get("final_confidence"),
            agent_agreement=final_state.get("agent_agreement"),
            errors=final_state.get("errors", []),
        )

        return final_state

    def analyze_batch(self, stocks: list[dict]) -> list[TradingState]:
        """
        Analyze multiple stocks sequentially.

        Note: This runs stocks one at a time (not parallel across stocks).
        Within each stock, the three agents still run in parallel.

        For true batch parallelism across stocks, use asyncio with ainvoke().
        That's a Sprint 3+ optimization — keeping it simple for now.

        Args:
            stocks: List of dicts with keys: symbol, market_data, indicators

        Returns:
            List of final TradingStates, one per stock
        """
        results = []
        for stock in stocks:
            try:
                result = self.analyze(
                    stock["symbol"],
                    stock["market_data"],
                    stock["indicators"],
                )
                results.append(result)
            except Exception as e:
                self.log.error("batch_item_failed", symbol=stock.get("symbol"), error=str(e))
        return results
