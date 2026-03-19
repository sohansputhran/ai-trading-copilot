"""
Pre-Trade Validator

Enforces hard limits before any trade is approved.

Design principle: This is a GATE, not advisory. A rejected trade does not
proceed. There is no override. This is non-negotiable by design.

Hard limits (from TECHNICAL_SPECS.md):
  - Max capital per trade:  5% of portfolio
  - Max daily loss:         2% of portfolio
  - Max open positions:     5 concurrent
  - Max sector exposure:    30% of portfolio

Why hard-code these as class constants rather than config?
  Risk limits are the last line of defense. Making them config means a
  misconfigured .env file or a typo could disable them. The 5/2/5/30
  values are business rules, not operational parameters. They belong in code.
  If they ever need to change, it's a deliberate code change + PR review.

What makes a good validator?
  1. Fast (no I/O, no LLM calls)
  2. Deterministic (same input → same result, every time)
  3. Explicit rejection reasons (never just "rejected")
  4. Composable (each check is independent, easy to add new checks)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import structlog

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Hard Limits — DO NOT make these environment variables
# ---------------------------------------------------------------------------
MAX_POSITION_PCT = 0.05          # 5%: max capital in any single trade
MAX_DAILY_LOSS_PCT = 0.02        # 2%: circuit breaker - stop trading for the day
MAX_OPEN_POSITIONS = 5           # hard cap on concurrent open trades
MAX_SECTOR_EXPOSURE_PCT = 0.30   # 30%: max portfolio in any one sector
MIN_CONFIDENCE_THRESHOLD = 0.60  # Matches aggregator's MIN_CONFIDENCE_TO_TRADE


@dataclass(frozen=True)
class ValidationResult:
    """
    Result of a pre-trade validation.

    frozen=True: downstream code cannot modify the verdict.

    Fields:
        approved:           True only if ALL checks pass
        rejection_reasons:  List of human-readable reasons (empty if approved)
        warnings:           Non-blocking observations (e.g., "position near max size")
        checks_passed:      How many individual checks passed
        checks_total:       Total checks run
    """
    approved: bool
    rejection_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks_passed: int = 0
    checks_total: int = 0

    @property
    def summary(self) -> str:
        if self.approved:
            return f"✅ APPROVED ({self.checks_passed}/{self.checks_total} checks passed)"
        reasons = "; ".join(self.rejection_reasons)
        return f"❌ REJECTED — {reasons}"


class PreTradeValidator:
    """
    Validates a proposed trade against all hard risk limits.

    This validator is stateless: it receives everything it needs to make
    a decision and returns a ValidationResult. No side effects.

    Usage:
        validator = PreTradeValidator()
        result = validator.validate(
            symbol="RELIANCE",
            position_size=PositionSize(...),
            portfolio=PortfolioSnapshot(...),
            confidence=0.78,
            daily_pnl=-3000.0,   # today's realized P&L so far
        )
        if not result.approved:
            print(result.summary)
    """

    def validate(
        self,
        symbol: str,
        position_value: float,
        portfolio_value: float,
        open_positions: int,
        confidence: float,
        daily_pnl: float = 0.0,
        sector: Optional[str] = None,
        sector_exposure: float = 0.0,
        capital_at_risk: float = 0.0,
    ) -> ValidationResult:
        """
        Run all pre-trade checks. Returns a ValidationResult.

        Args:
            symbol:           Stock symbol being traded
            position_value:   Total cost of the proposed position
            portfolio_value:  Current total portfolio value
            open_positions:   Number of currently open trades
            confidence:       Multi-agent confidence score (0-1)
            daily_pnl:        Today's realized P&L (negative = loss)
            sector:           Sector of the stock (optional, for sector check)
            sector_exposure:  Current portfolio exposure to this sector (rupees)
            capital_at_risk:  Dollar amount at risk on this trade (position - stop)

        Returns:
            ValidationResult with approved=True only if all hard limits pass
        """
        rejections: list[str] = []
        warnings: list[str] = []
        checks_passed = 0
        checks_total = 0

        # ------------------------------------------------------------------
        # Check 1: Position size limit (5% of portfolio)
        # ------------------------------------------------------------------
        checks_total += 1
        position_pct = position_value / portfolio_value if portfolio_value > 0 else 1.0
        if position_pct > MAX_POSITION_PCT:
            rejections.append(
                f"Position size {position_pct * 100:.1f}% exceeds max "
                f"{MAX_POSITION_PCT * 100:.0f}% of portfolio "
                f"({position_value:,.0f} > {portfolio_value * MAX_POSITION_PCT:,.0f})"
            )
        else:
            checks_passed += 1
            if position_pct > MAX_POSITION_PCT * 0.80:  # within 80% of limit
                warnings.append(
                    f"Position size {position_pct * 100:.1f}% is close to "
                    f"the {MAX_POSITION_PCT * 100:.0f}% limit"
                )

        # ------------------------------------------------------------------
        # Check 2: Daily loss circuit breaker (2% of portfolio)
        # ------------------------------------------------------------------
        checks_total += 1
        daily_loss_pct = abs(daily_pnl) / portfolio_value if portfolio_value > 0 else 0
        if daily_pnl < 0 and daily_loss_pct >= MAX_DAILY_LOSS_PCT:
            rejections.append(
                f"Daily loss circuit breaker triggered: today's loss "
                f"{abs(daily_pnl):,.0f} ({daily_loss_pct * 100:.1f}%) "
                f"has reached the {MAX_DAILY_LOSS_PCT * 100:.0f}% daily limit. "
                f"No new trades for the rest of the day."
            )
        else:
            checks_passed += 1
            if daily_pnl < 0 and daily_loss_pct > MAX_DAILY_LOSS_PCT * 0.75:
                warnings.append(
                    f"Daily loss {daily_loss_pct * 100:.1f}% is approaching "
                    f"the {MAX_DAILY_LOSS_PCT * 100:.0f}% daily circuit breaker"
                )

        # ------------------------------------------------------------------
        # Check 3: Max open positions (5 concurrent)
        # ------------------------------------------------------------------
        checks_total += 1
        if open_positions >= MAX_OPEN_POSITIONS:
            rejections.append(
                f"Max open positions limit reached: {open_positions}/{MAX_OPEN_POSITIONS}. "
                f"Close an existing position before opening a new one."
            )
        else:
            checks_passed += 1
            if open_positions == MAX_OPEN_POSITIONS - 1:
                warnings.append(
                    f"This would be position {open_positions + 1}/{MAX_OPEN_POSITIONS} "
                    f"(maximum). One slot remaining after this trade."
                )

        # ------------------------------------------------------------------
        # Check 4: Sector exposure limit (30% of portfolio)
        # ------------------------------------------------------------------
        checks_total += 1
        if sector is not None:
            new_sector_exposure = sector_exposure + position_value
            new_sector_pct = new_sector_exposure / portfolio_value if portfolio_value > 0 else 1.0
            if new_sector_pct > MAX_SECTOR_EXPOSURE_PCT:
                rejections.append(
                    f"Sector exposure limit exceeded: adding this trade would bring "
                    f"{sector} sector to {new_sector_pct * 100:.1f}% of portfolio "
                    f"(limit: {MAX_SECTOR_EXPOSURE_PCT * 100:.0f}%)"
                )
            else:
                checks_passed += 1
                if new_sector_pct > MAX_SECTOR_EXPOSURE_PCT * 0.80:
                    warnings.append(
                        f"{sector} sector exposure would reach {new_sector_pct * 100:.1f}% "
                        f"(approaching {MAX_SECTOR_EXPOSURE_PCT * 100:.0f}% limit)"
                    )
        else:
            # Sector unknown — pass the check but warn
            checks_passed += 1
            warnings.append("Sector unknown — sector exposure check skipped")

        # ------------------------------------------------------------------
        # Check 5: Minimum confidence threshold
        # ------------------------------------------------------------------
        checks_total += 1
        if confidence < MIN_CONFIDENCE_THRESHOLD:
            rejections.append(
                f"Agent confidence {confidence:.2f} below minimum threshold "
                f"{MIN_CONFIDENCE_THRESHOLD:.2f}. Signal is too uncertain to trade."
            )
        else:
            checks_passed += 1

        # ------------------------------------------------------------------
        # Check 6: Portfolio not empty (sanity check)
        # ------------------------------------------------------------------
        checks_total += 1
        if portfolio_value <= 0:
            rejections.append("Portfolio value is zero or negative — cannot size position.")
        else:
            checks_passed += 1

        approved = len(rejections) == 0

        result = ValidationResult(
            approved=approved,
            rejection_reasons=rejections,
            warnings=warnings,
            checks_passed=checks_passed,
            checks_total=checks_total,
        )

        logger.info(
            "pre_trade_validation",
            symbol=symbol,
            approved=approved,
            checks_passed=checks_passed,
            checks_total=checks_total,
            rejection_count=len(rejections),
            warning_count=len(warnings),
        )

        if not approved:
            for reason in rejections:
                logger.warning("trade_rejected", symbol=symbol, reason=reason)

        return result
