"""
Position Sizer

Answers: "How many shares should I buy?"

Three sizing methods:
  - KELLY:           fraction of capital based on Kelly Criterion (half-Kelly by default)
  - FIXED_FRACTIONAL: fixed % of portfolio risked per trade
  - ATR_BASED:       position sized so that 1 ATR move = fixed dollar risk

Why separate methods?
  - Kelly needs historical win rate data (calibrated over 100+ trades)
  - Fixed fractional is simpler and more robust when starting out
  - ATR-based adapts to current volatility — preferred in production

Design principle: PositionSize is a pure value object (no side effects).
The sizer reads config from environment variables so it can be tuned without
code changes — a 12-factor app principle.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum

import structlog

logger = structlog.get_logger()


class SizingMethod(StrEnum):
    """Which position sizing algorithm to use."""
    KELLY = "kelly"
    FIXED_FRACTIONAL = "fixed_fractional"
    ATR_BASED = "atr_based"


@dataclass(frozen=True)
class PositionSize:
    """
    Result of a position sizing calculation.

    Frozen (immutable) so downstream code can't accidentally mutate it.

    Fields:
        shares:          Number of shares to buy (always a whole number)
        capital_at_risk: Dollar amount risked on this trade (entry - stop) * shares
        position_value:  Total cost to open this position (entry_price * shares)
        fraction_used:   What fraction of portfolio this position represents
        method:          Which sizing method produced this result
        reasoning:       Human-readable explanation (useful for dashboard + journal)
    """
    shares: int
    capital_at_risk: float
    position_value: float
    fraction_used: float
    method: SizingMethod
    reasoning: str


class PositionSizer:
    """
    Calculates position sizes using configurable algorithms.

    Configuration (via environment variables):
        SIZING_METHOD:        "kelly" | "fixed_fractional" | "atr_based"  (default: fixed_fractional)
        RISK_PER_TRADE_PCT:   % of portfolio to risk per trade             (default: 1.0)
        KELLY_FRACTION:       fraction of full Kelly to use                (default: 0.5 = half-Kelly)
        HISTORICAL_WIN_RATE:  estimated win rate for Kelly calc            (default: 0.55)
        ATR_RISK_MULTIPLIER:  how many ATRs = 1R of risk for ATR method   (default: 1.5)
        MAX_POSITION_PCT:     hard cap: no position > this % of portfolio  (default: 5.0)

    Usage:
        sizer = PositionSizer(portfolio_value=500_000)
        size = sizer.calculate(
            entry_price=2500.0,
            stop_loss=2450.0,
            atr=35.0,
            confidence=0.78,
        )
        print(size.shares, size.reasoning)
    """

    # Hard cap from TECHNICAL_SPECS.md - cannot be exceeded regardless of method
    ABSOLUTE_MAX_POSITION_PCT = 0.05  # 5%

    def __init__(self, portfolio_value: float) -> None:
        if portfolio_value <= 0:
            raise ValueError(f"Portfolio value must be positive, got {portfolio_value}")

        self.portfolio_value = portfolio_value

        # Load config from environment, fall back to safe defaults
        method_str = os.getenv("SIZING_METHOD", "fixed_fractional").lower()
        try:
            self.method = SizingMethod(method_str)
        except ValueError:
            logger.warning(
                "unknown_sizing_method",
                method=method_str,
                fallback="fixed_fractional",
            )
            self.method = SizingMethod.FIXED_FRACTIONAL

        self.risk_per_trade_pct = float(os.getenv("RISK_PER_TRADE_PCT", "1.0")) / 100
        self.kelly_fraction = float(os.getenv("KELLY_FRACTION", "0.5"))
        self.historical_win_rate = float(os.getenv("HISTORICAL_WIN_RATE", "0.55"))
        self.atr_risk_multiplier = float(os.getenv("ATR_RISK_MULTIPLIER", "1.5"))

        logger.info(
            "position_sizer_initialized",
            method=self.method,
            portfolio_value=portfolio_value,
            risk_per_trade_pct=self.risk_per_trade_pct * 100,
        )

    def calculate(
        self,
        entry_price: float,
        stop_loss: float,
        atr: float | None = None,
        confidence: float = 0.5,
        reward_risk_ratio: float = 2.0,
    ) -> PositionSize:
        """
        Calculate position size using the configured method.

        Args:
            entry_price:      Price at which we'd enter the trade
            stop_loss:        Price at which we'd exit if wrong (defines 1R)
            atr:              Average True Range (required for ATR_BASED method)
            confidence:       Agent confidence score 0-1 (used to scale Kelly)
            reward_risk_ratio: Expected reward / risk (used in Kelly formula)

        Returns:
            PositionSize with shares, capital at risk, and reasoning

        Note on stop_loss: entry must be above stop for long trades.
        We enforce this defensively — if stop >= entry, we use a minimum
        risk of 0.5% of entry price to avoid division by zero.
        """
        # Defensive: ensure stop is below entry (long-only for now)
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            logger.warning(
                "invalid_stop_loss",
                entry=entry_price,
                stop=stop_loss,
                action="using_minimum_risk",
            )
            risk_per_share = entry_price * 0.005  # 0.5% minimum risk

        if self.method == SizingMethod.KELLY:
            return self._kelly_size(entry_price, risk_per_share, confidence, reward_risk_ratio)
        elif self.method == SizingMethod.ATR_BASED:
            return self._atr_size(entry_price, risk_per_share, atr)
        else:  # default: FIXED_FRACTIONAL
            return self._fixed_fractional_size(entry_price, risk_per_share)

    # -------------------------------------------------------------------------
    # Private sizing implementations
    # -------------------------------------------------------------------------

    def _fixed_fractional_size(self, entry_price: float, risk_per_share: float) -> PositionSize:
        """
        Fixed Fractional Sizing.

        Risk a fixed percentage of portfolio on every trade, regardless of edge.

        capital_to_risk = portfolio_value × risk_per_trade_pct
        shares = capital_to_risk / risk_per_share

        Example:
            Portfolio = 500,000, risk_pct = 1%, entry = 2500, stop = 2450
            capital_to_risk = 5,000
            risk_per_share  = 50
            shares          = 100
        """
        capital_to_risk = self.portfolio_value * self.risk_per_trade_pct
        raw_shares = capital_to_risk / risk_per_share
        shares = self._apply_max_cap(raw_shares, entry_price)

        capital_at_risk = shares * risk_per_share
        position_value = shares * entry_price
        fraction_used = position_value / self.portfolio_value

        reasoning = (
            f"Fixed fractional sizing: risking {self.risk_per_trade_pct * 100:.1f}% "
            f"of {self.portfolio_value:,.0f} = {capital_to_risk:,.0f}. "
            f"Risk per share = {risk_per_share:.2f} → {shares} shares "
            f"({position_value:,.0f} position, {fraction_used * 100:.1f}% of portfolio)."
        )

        logger.info(
            "position_sized_fixed_fractional",
            shares=shares,
            capital_at_risk=round(capital_at_risk, 2),
            position_value=round(position_value, 2),
        )

        return PositionSize(
            shares=shares,
            capital_at_risk=round(capital_at_risk, 2),
            position_value=round(position_value, 2),
            fraction_used=round(fraction_used, 4),
            method=SizingMethod.FIXED_FRACTIONAL,
            reasoning=reasoning,
        )

    def _kelly_size(
        self,
        entry_price: float,
        risk_per_share: float,
        confidence: float,
        reward_risk_ratio: float,
    ) -> PositionSize:
        """
        Half-Kelly Sizing (scaled by agent confidence).

        Formula: f* = (b*p - q) / b
          where b = reward_risk_ratio, p = win_rate, q = 1 - p

        We then apply two modifiers:
          1. kelly_fraction (default 0.5): shrinks full Kelly to half-Kelly
          2. confidence: agent confidence scales the fraction further
             e.g., 0.78 confidence * half-Kelly = more conservative than 1.0 confidence

        This means a high-confidence agent signal gets more capital than a low-confidence one.
        That's the key advantage of Kelly for ML/AI-driven systems.

        Why scale by confidence?
            Traditional Kelly uses a fixed historical win rate. But our agents produce
            a per-trade confidence score. We treat confidence as our best estimate of
            the probability that this specific trade will be a winner, blending it
            with historical win rate for robustness.
        """
        p = (self.historical_win_rate + confidence) / 2  # blend historical + per-trade
        q = 1.0 - p
        b = reward_risk_ratio

        if b <= 0:
            logger.warning("invalid_reward_risk_ratio", b=b, fallback="fixed_fractional")
            return self._fixed_fractional_size(entry_price, risk_per_share)

        full_kelly = (b * p - q) / b

        if full_kelly <= 0:
            # Negative Kelly = no edge. Fall back to minimum fixed fractional.
            logger.info(
                "kelly_no_edge",
                full_kelly=round(full_kelly, 4),
                p=round(p, 3),
                b=b,
                action="fallback_to_fixed_fractional",
            )
            return self._fixed_fractional_size(entry_price, risk_per_share)

        adjusted_kelly = full_kelly * self.kelly_fraction * confidence
        # Never exceed the hard cap
        adjusted_kelly = min(adjusted_kelly, self.ABSOLUTE_MAX_POSITION_PCT)

        capital_to_risk = self.portfolio_value * adjusted_kelly
        raw_shares = capital_to_risk / risk_per_share
        shares = self._apply_max_cap(raw_shares, entry_price)

        capital_at_risk = shares * risk_per_share
        position_value = shares * entry_price
        fraction_used = position_value / self.portfolio_value

        reasoning = (
            f"Kelly sizing: full Kelly={full_kelly * 100:.1f}%, "
            f"half-Kelly×confidence={adjusted_kelly * 100:.2f}%. "
            f"Win prob={p:.2f} (blend of historical {self.historical_win_rate:.2f} "
            f"+ confidence {confidence:.2f}), b={b:.1f}. "
            f"→ {shares} shares, {capital_at_risk:,.0f} at risk "
            f"({fraction_used * 100:.1f}% of portfolio)."
        )

        logger.info(
            "position_sized_kelly",
            full_kelly=round(full_kelly, 4),
            adjusted_kelly=round(adjusted_kelly, 4),
            shares=shares,
            capital_at_risk=round(capital_at_risk, 2),
        )

        return PositionSize(
            shares=shares,
            capital_at_risk=round(capital_at_risk, 2),
            position_value=round(position_value, 2),
            fraction_used=round(fraction_used, 4),
            method=SizingMethod.KELLY,
            reasoning=reasoning,
        )

    def _atr_size(
        self,
        entry_price: float,
        risk_per_share: float,
        atr: float | None,
    ) -> PositionSize:
        """
        ATR-Based Sizing.

        Goal: keep dollar risk constant regardless of the stock's price or volatility.

        stop_distance = atr × atr_risk_multiplier  (e.g., 1.5 ATR = your stop)
        dollar_risk   = portfolio_value × risk_per_trade_pct
        shares        = dollar_risk / stop_distance

        Why ATR for stop distance?
            A 50 stock moving 50 paise is very different from a 5000 stock moving
            50. ATR normalizes for volatility — your stop is placed at a distance
            proportional to how much the stock actually moves day-to-day.

        If atr is not provided, falls back to fixed fractional.
        """
        if atr is None or atr <= 0:
            logger.warning("atr_not_available", action="fallback_to_fixed_fractional")
            return self._fixed_fractional_size(entry_price, risk_per_share)

        stop_distance = atr * self.atr_risk_multiplier
        dollar_risk = self.portfolio_value * self.risk_per_trade_pct
        raw_shares = dollar_risk / stop_distance
        shares = self._apply_max_cap(raw_shares, entry_price)

        capital_at_risk = shares * stop_distance
        position_value = shares * entry_price
        fraction_used = position_value / self.portfolio_value

        reasoning = (
            f"ATR-based sizing: ATR={atr:.2f} x {self.atr_risk_multiplier}x = "
            f"{stop_distance:.2f} stop distance. "
            f"Dollar risk target = {dollar_risk:,.0f} → {shares} shares "
            f"({position_value:,.0f} position, {fraction_used * 100:.1f}% of portfolio)."
        )

        logger.info(
            "position_sized_atr",
            atr=atr,
            stop_distance=round(stop_distance, 2),
            shares=shares,
            capital_at_risk=round(capital_at_risk, 2),
        )

        return PositionSize(
            shares=shares,
            capital_at_risk=round(capital_at_risk, 2),
            position_value=round(position_value, 2),
            fraction_used=round(fraction_used, 4),
            method=SizingMethod.ATR_BASED,
            reasoning=reasoning,
        )

    def _apply_max_cap(self, raw_shares: float, entry_price: float) -> int:
        """
        Enforce the hard cap: no position > ABSOLUTE_MAX_POSITION_PCT of portfolio.
        Returns whole shares (floor, never round up into extra risk).
        """
        max_capital = self.portfolio_value * self.ABSOLUTE_MAX_POSITION_PCT
        max_shares_by_cap = max_capital / entry_price if entry_price > 0 else 0
        capped_shares = min(raw_shares, max_shares_by_cap)

        if capped_shares < raw_shares:
            logger.info(
                "position_capped_at_max",
                raw_shares=int(raw_shares),
                capped_shares=int(capped_shares),
                max_pct=self.ABSOLUTE_MAX_POSITION_PCT * 100,
            )

        return max(0, int(capped_shares))  # floor + guard against negatives
