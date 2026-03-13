"""
Base Agent
===========
Abstract base class that every strategy agent inherits from.

Why a base class?
- Enforces a consistent interface across all agents
- Puts the HuggingFace call + signal extraction logic in ONE place
- Individual agents only need to define: their prompt, which indicators
  they care about, and how to compute confidence
- Makes it trivial to add a 4th or 5th agent later

The pattern each agent follows:
    1. extract_relevant_indicators() → pull the subset of indicators it cares about
    2. build_prompt()               → format a focused system + user prompt
    3. call_llm()                   → HuggingFace InferenceClient call (defined here)
    4. parse_signal()               → extract Signal enum from free-text LLM output
    5. compute_confidence()         → rule-based confidence from indicator values
    6. return AgentAnalysis         → typed, validated output

HuggingFace integration:
    Uses InferenceClient with chat_completion() — same as Sprint 1's scanner_agent.py.
    Pass the already-initialized InferenceClient from scanner_agent into each agent.
    If llm_client is None, agents fall back to pure rule-based analysis.
"""

import re
import structlog
from abc import ABC, abstractmethod
from typing import Dict, Any

from src.agents.state import TradingState, AgentAnalysis, Signal

logger = structlog.get_logger()


class BaseStrategyAgent(ABC):
    """
    Abstract base for all strategy agents.

    Subclasses must implement:
    - agent_name: str
    - analyze(state) -> AgentAnalysis
    """

    agent_name: str = "base"

    def __init__(self, llm_client=None, llm_model: str = "unknown"):
        """
        Args:
            llm_client:  huggingface_hub.InferenceClient instance — the same object
                         created in scanner_agent.py. Reusing it means one
                         connectivity test, one token, shared across all agents.
            llm_model:   Model name string for logging (e.g. "Llama-3-8B").
        """
        self.llm_client = llm_client
        self.llm_model  = llm_model
        self.log        = logger.bind(agent=self.agent_name)

    @abstractmethod
    def analyze(self, state: TradingState) -> AgentAnalysis:
        """
        Main entry point. Takes the full state, returns an AgentAnalysis.
        Each subclass implements its own strategy logic here.
        """
        ...

    # ─────────────────────────────────────────────
    # Shared LLM utilities (used by all subclasses)
    # ─────────────────────────────────────────────

    def call_llm(self, system_prompt: str, user_message: str, max_tokens: int = 200) -> str:
        """
        Call HuggingFace InferenceClient with chat_completion().

        Mirrors the exact pattern from Sprint 1's scanner_agent.py so the
        same InferenceClient object works seamlessly across the whole system.

        Returns generated text, or empty string if LLM is unavailable.
        Empty string causes subclasses to use their rule-based reasoning text.
        """
        if self.llm_client is None:
            self.log.debug("llm_not_configured_using_rules")
            return ""

        try:
            self.log.info("llm_call_start", model=self.llm_model)
            response = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=0.3,   # Low temperature = consistent signals
            )
            generated = response.choices[0].message.content.strip()
            self.log.info("llm_call_complete", output_length=len(generated))
            return generated

        except Exception as e:
            self.log.warning("llm_call_failed", error=str(e))
            return ""

    def parse_signal_from_text(self, text: str) -> Signal:
        """
        Extract BUY / SELL / HOLD from free-form LLM text.

        Why this approach instead of asking LLM to return JSON?
        - Small HuggingFace models often fail to produce valid JSON
        - Free-form text reasoning is what these models do well
        - We do the structured extraction ourselves (more reliable)

        Strategy: look for signal keywords in order of specificity.
        "strong buy" → BUY, "do not buy" → HOLD, "buy" → BUY
        """
        text_lower = text.lower()

        # Negative patterns first (order matters — "do not buy" should → HOLD)
        negative_patterns = [
            r"do not buy", r"don't buy", r"avoid buying",
            r"do not sell", r"don't sell",
            r"not a buy", r"not a sell",
        ]
        for pattern in negative_patterns:
            if re.search(pattern, text_lower):
                return Signal.HOLD

        buy_patterns  = [r"\bbuy\b", r"\bbullish\b", r"\blong\b", r"\bupside\b", r"\bbreakout\b"]
        sell_patterns = [r"\bsell\b", r"\bbearish\b", r"\bshort\b", r"\bdownside\b", r"\bbreakdown\b"]

        buy_count  = sum(1 for p in buy_patterns  if re.search(p, text_lower))
        sell_count = sum(1 for p in sell_patterns if re.search(p, text_lower))

        if buy_count > sell_count:
            return Signal.BUY
        elif sell_count > buy_count:
            return Signal.SELL
        else:
            return Signal.HOLD

    def safe_get(self, indicators: Dict[str, Any], key: str, default: float = 0.0) -> float:
        """
        Safely retrieve an indicator value with a fallback default.

        Prevents KeyError if an indicator wasn't computed (e.g. insufficient data).
        Logs a warning so you know which indicators are missing in production.
        """
        value = indicators.get(key, None)
        if value is None:
            self.log.warning("indicator_missing", indicator=key, using_default=default)
            return default
        return float(value)
