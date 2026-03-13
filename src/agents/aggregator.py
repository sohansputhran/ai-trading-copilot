"""
Aggregator
===========
Combines outputs from all three strategy agents into a final trading decision.

Key design decisions:
1. Rule-based, NOT an LLM — aggregation is math, not language
2. Weighted voting — not all agents are equal for all market conditions
3. Agreement penalty — low consensus = lower confidence (uncertainty is information)
4. Transparent reasoning — final_reasoning explains HOW the decision was made

Agent weights (configurable via configs/agents.yaml):
    Technical Analysis : 0.35  → Most reliable for entry timing
    Momentum Strategy  : 0.35  → Strong predictor for trend continuation
    Breakout Strategy  : 0.30  → High signal quality but more false positives

Confidence thresholds (what triggers a BUY/SELL recommendation):
    final_confidence >= 0.65 AND agent_agreement >= 0.67 (2/3 agents agree)
    Otherwise → HOLD (don't trade uncertain signals)
"""

import structlog
from typing import List

from src.agents.state import TradingState, AgentAnalysis, Signal, AgentName

logger = structlog.get_logger()

# ── Configurable weights (sum must equal 1.0) ─────────────────────────────────
AGENT_WEIGHTS = {
    AgentName.TECHNICAL: 0.35,
    AgentName.MOMENTUM:  0.35,
    AgentName.BREAKOUT:  0.30,
}

# ── Thresholds ─────────────────────────────────────────────────────────────────
MIN_CONFIDENCE_TO_TRADE = 0.60   # Below this → always HOLD
MIN_AGREEMENT_TO_TRADE  = 0.67   # Below this (< 2/3 agents agree) → always HOLD


def aggregate(state: TradingState) -> TradingState:
    """
    LangGraph node function for the aggregator.

    Reads all three agent analyses from state.
    Writes final_signal, final_confidence, final_reasoning, agent_agreement.

    This function is intentionally pure — it only reads from state and
    returns a dict of updates. LangGraph merges this dict into state.
    """
    log = logger.bind(symbol=state["symbol"], node="aggregator")
    log.info("aggregator_start")

    analyses = _collect_analyses(state)

    if not analyses:
        log.error("no_agent_analyses_available")
        return {
            "final_signal":     Signal.HOLD,
            "final_confidence": 0.0,
            "final_reasoning":  "No agent analyses available. Defaulting to HOLD.",
            "agent_agreement":  0.0,
            "errors": state["errors"] + ["Aggregator: no analyses to aggregate"],
        }

    # ── Step 1: Compute agreement ─────────────────────────────────────────────
    signals     = [a.signal for a in analyses]
    buy_count   = signals.count(Signal.BUY)
    sell_count  = signals.count(Signal.SELL)
    hold_count  = signals.count(Signal.HOLD)
    total       = len(analyses)

    dominant_count  = max(buy_count, sell_count, hold_count)
    agent_agreement = dominant_count / total  # 1.0 = unanimous, 0.33 = all different

    # ── Step 2: Determine dominant signal ─────────────────────────────────────
    if buy_count > sell_count and buy_count > hold_count:
        dominant_signal = Signal.BUY
    elif sell_count > buy_count and sell_count > hold_count:
        dominant_signal = Signal.SELL
    else:
        dominant_signal = Signal.HOLD

    # ── Step 3: Compute weighted confidence ───────────────────────────────────
    weighted_confidence = _weighted_confidence(analyses, dominant_signal)

    # ── Step 4: Apply agreement penalty ───────────────────────────────────────
    # If agents disagree significantly, reduce our confidence in the signal.
    # Example: 2 BUY (conf 0.8) + 1 SELL (conf 0.7) → agreement=0.67
    # Without penalty: high confidence. With penalty: we acknowledge uncertainty.
    agreement_penalty   = 1.0 - (0.3 * (1.0 - agent_agreement))
    final_confidence    = round(weighted_confidence * agreement_penalty, 3)

    # ── Step 5: Apply trading thresholds ──────────────────────────────────────
    if final_confidence < MIN_CONFIDENCE_TO_TRADE or agent_agreement < MIN_AGREEMENT_TO_TRADE:
        final_signal = Signal.HOLD
        threshold_note = (
            f" [Overridden to HOLD: confidence={final_confidence:.2f} "
            f"(min={MIN_CONFIDENCE_TO_TRADE}), "
            f"agreement={agent_agreement:.2f} (min={MIN_AGREEMENT_TO_TRADE})]"
        )
    else:
        final_signal  = dominant_signal
        threshold_note = ""

    # ── Step 6: Build human-readable reasoning ────────────────────────────────
    final_reasoning = _build_reasoning(
        analyses, dominant_signal, final_signal,
        final_confidence, agent_agreement, threshold_note
    )

    log.info(
        "aggregator_complete",
        final_signal=final_signal,
        final_confidence=final_confidence,
        agent_agreement=agent_agreement,
        buy_count=buy_count,
        sell_count=sell_count,
        hold_count=hold_count,
    )

    return {
        "final_signal":     final_signal,
        "final_confidence": final_confidence,
        "final_reasoning":  final_reasoning,
        "agent_agreement":  agent_agreement,
    }


# ─────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────

def _collect_analyses(state: TradingState) -> List[AgentAnalysis]:
    """Pull all non-None agent analyses from state."""
    analyses = []
    for key in ("technical_analysis", "momentum_analysis", "breakout_analysis"):
        analysis = state.get(key)
        if analysis is not None:
            analyses.append(analysis)
    return analyses


def _weighted_confidence(analyses: List[AgentAnalysis], target_signal: Signal) -> float:
    """
    Compute confidence as a weighted average, but only for agents
    that agree with the dominant signal.

    Why only agreeing agents?
    - Disagreeing agents' confidence is captured by the agreement score
    - Including them in the confidence calculation would be double-penalizing
    - A dissenting agent with high confidence already reduces agreement score
    """
    total_weight = 0.0
    weighted_sum = 0.0

    for analysis in analyses:
        agent_key = AgentName(analysis.agent_name)
        weight    = AGENT_WEIGHTS.get(agent_key, 1.0 / len(analyses))

        if analysis.signal == target_signal:
            weighted_sum   += weight * analysis.confidence
            total_weight   += weight

    if total_weight == 0:
        return 0.0

    return weighted_sum / total_weight


def _build_reasoning(
    analyses: List[AgentAnalysis],
    dominant_signal: Signal,
    final_signal: Signal,
    final_confidence: float,
    agent_agreement: float,
    threshold_note: str,
) -> str:
    """Build a clear, structured explanation of the final decision."""
    lines = [
        f"FINAL DECISION: {final_signal.value} (confidence: {final_confidence:.0%}, "
        f"agent agreement: {agent_agreement:.0%}){threshold_note}",
        "",
        "AGENT BREAKDOWN:",
    ]

    for analysis in analyses:
        agreement_marker = "✓" if analysis.signal == dominant_signal else "✗"
        lines.append(
            f"  {agreement_marker} {analysis.agent_name.upper()}: "
            f"{analysis.signal.value} ({analysis.confidence:.0%}) — {analysis.reasoning[:120]}..."
        )

    if any(a.warnings for a in analyses):
        lines.append("")
        lines.append("WARNINGS:")
        for analysis in analyses:
            for warning in analysis.warnings:
                lines.append(f"  ⚠ [{analysis.agent_name}] {warning}")

    return "\n".join(lines)
