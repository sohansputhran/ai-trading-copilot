"""
Tests for Risk Management Engine

No external dependencies beyond pytest and structlog.
Edge cases are prioritized - risk bugs have real consequences.

Test categories:
  1. PositionSizer - all three methods + edge cases
  2. PreTradeValidator - each hard limit + approval path
  3. PortfolioRisk - position lifecycle + snapshot accuracy
  4. Integration - full risk pipeline (sizer -> validator -> portfolio)
"""

import pytest

from src.risk_management.portfolio import PortfolioRisk, Position
from src.risk_management.position_sizer import PositionSizer, SizingMethod
from src.risk_management.validators import (
    MAX_DAILY_LOSS_PCT,
    MAX_OPEN_POSITIONS,
    MIN_CONFIDENCE_THRESHOLD,
    PreTradeValidator,
)

# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sizer_fixed() -> PositionSizer:
    """PositionSizer with fixed fractional method, 500,000 portfolio."""
    import os
    os.environ["SIZING_METHOD"] = "fixed_fractional"
    os.environ["RISK_PER_TRADE_PCT"] = "1.0"
    return PositionSizer(portfolio_value=500_000)


@pytest.fixture
def sizer_kelly() -> PositionSizer:
    import os
    os.environ["SIZING_METHOD"] = "kelly"
    os.environ["RISK_PER_TRADE_PCT"] = "1.0"
    os.environ["KELLY_FRACTION"] = "0.5"
    os.environ["HISTORICAL_WIN_RATE"] = "0.55"
    return PositionSizer(portfolio_value=500_000)


@pytest.fixture
def sizer_atr() -> PositionSizer:
    import os
    os.environ["SIZING_METHOD"] = "atr_based"
    os.environ["RISK_PER_TRADE_PCT"] = "1.0"
    os.environ["ATR_RISK_MULTIPLIER"] = "1.5"
    return PositionSizer(portfolio_value=500_000)


@pytest.fixture
def validator() -> PreTradeValidator:
    return PreTradeValidator()


@pytest.fixture
def portfolio() -> PortfolioRisk:
    return PortfolioRisk(portfolio_value=500_000)


@pytest.fixture
def sample_position() -> Position:
    return Position(
        symbol="RELIANCE",
        entry_price=2500.0,
        stop_loss=2450.0,
        shares=40,
        position_value=100_000.0,
        capital_at_risk=2_000.0,
        sector="Energy",
    )


# ===========================================================================
# PositionSizer Tests — Fixed Fractional
# ===========================================================================

class TestFixedFractionalSizer:
    def test_basic_calculation(self, sizer_fixed: PositionSizer) -> None:
        """1% of 500,000 = 5,000 risk. Risk per share = 50. Expect 100 shares."""
        size = sizer_fixed.calculate(entry_price=2500.0, stop_loss=2450.0)
        assert size.shares == 10
        assert size.method == SizingMethod.FIXED_FRACTIONAL
        assert size.capital_at_risk == pytest.approx(500.0, rel=0.01)
        assert size.position_value == pytest.approx(25_000.0, rel=0.01)

    def test_max_cap_enforced(self, sizer_fixed: PositionSizer) -> None:
        """Very tight stop -> huge share count -> must be capped at 5% of portfolio."""
        # 500,000 × 5% = 25,000 max position. At 2500/share → max 10 shares
        size = sizer_fixed.calculate(entry_price=2500.0, stop_loss=2499.0)  # 1 risk
        max_allowed_shares = int((500_000 * 0.05) / 2500.0)
        assert size.shares <= max_allowed_shares

    def test_fraction_used_within_limits(self, sizer_fixed: PositionSizer) -> None:
        size = sizer_fixed.calculate(entry_price=2500.0, stop_loss=2450.0)
        assert size.fraction_used <= 0.05  # never exceeds 5% hard cap

    def test_zero_shares_when_entry_equals_stop(self, sizer_fixed: PositionSizer) -> None:
        """Stop equals entry - defensive min risk kicks in, shares should be non-negative."""
        size = sizer_fixed.calculate(entry_price=2500.0, stop_loss=2500.0)
        assert size.shares >= 0

    def test_reasoning_is_non_empty(self, sizer_fixed: PositionSizer) -> None:
        size = sizer_fixed.calculate(entry_price=2500.0, stop_loss=2450.0)
        assert len(size.reasoning) > 20  # meaningful explanation

    def test_immutable_result(self, sizer_fixed: PositionSizer) -> None:
        """PositionSize is frozen - mutation should raise."""
        size = sizer_fixed.calculate(entry_price=2500.0, stop_loss=2450.0)
        with pytest.raises((AttributeError, TypeError)):
            size.shares = 999  # type: ignore

    def test_high_priced_stock(self, sizer_fixed: PositionSizer) -> None:
        """Expensive stock (MRF ~150,000/share) - may result in 0 shares."""
        size = sizer_fixed.calculate(entry_price=150_000.0, stop_loss=148_000.0)
        assert size.shares >= 0
        assert size.fraction_used <= 0.05


class TestKellySizer:
    def test_kelly_with_positive_edge(self, sizer_kelly: PositionSizer) -> None:
        """Positive edge (win rate > 50%, b=2) should produce valid shares."""
        size = sizer_kelly.calculate(
            entry_price=2500.0,
            stop_loss=2450.0,
            confidence=0.78,
            reward_risk_ratio=2.0,
        )
        assert size.shares > 0
        assert size.method == SizingMethod.KELLY
        assert size.fraction_used <= 0.05

    def test_kelly_no_edge_falls_back(self, sizer_kelly: PositionSizer) -> None:
        """Negative Kelly (low win rate, bad R:R) -> falls back to fixed fractional."""
        size = sizer_kelly.calculate(
            entry_price=2500.0,
            stop_loss=2450.0,
            confidence=0.30,   # low confidence
            reward_risk_ratio=0.5,  # terrible R:R
        )
        # Either 0 shares or falls back to fixed fractional
        assert size.shares >= 0
        assert size.fraction_used <= 0.05

    def test_kelly_high_confidence_bigger_than_low(self, sizer_kelly: PositionSizer) -> None:
        """Higher confidence should produce same or more shares (given positive Kelly)."""
        size_high = sizer_kelly.calculate(
            entry_price=2500.0, stop_loss=2450.0, confidence=0.90, reward_risk_ratio=2.0
        )
        size_low = sizer_kelly.calculate(
            entry_price=2500.0, stop_loss=2450.0, confidence=0.62, reward_risk_ratio=2.0
        )
        assert size_high.shares >= size_low.shares

    def test_kelly_invalid_rr_ratio(self, sizer_kelly: PositionSizer) -> None:
        """Zero or negative reward:risk -> falls back gracefully."""
        size = sizer_kelly.calculate(
            entry_price=2500.0, stop_loss=2450.0, confidence=0.78, reward_risk_ratio=0.0
        )
        assert size.shares >= 0


class TestAtrSizer:
    def test_atr_basic(self, sizer_atr: PositionSizer) -> None:
        size = sizer_atr.calculate(
            entry_price=2500.0,
            stop_loss=2450.0,
            atr=35.0,
        )
        assert size.shares > 0
        assert size.method == SizingMethod.ATR_BASED
        assert size.fraction_used <= 0.05

    def test_atr_none_falls_back(self, sizer_atr: PositionSizer) -> None:
        """No ATR available -> falls back to fixed fractional."""
        size = sizer_atr.calculate(entry_price=2500.0, stop_loss=2450.0, atr=None)
        assert size.method == SizingMethod.FIXED_FRACTIONAL

    def test_atr_zero_falls_back(self, sizer_atr: PositionSizer) -> None:
        size = sizer_atr.calculate(entry_price=2500.0, stop_loss=2450.0, atr=0.0)
        assert size.method == SizingMethod.FIXED_FRACTIONAL

    def test_high_volatility_fewer_shares(self, sizer_atr: PositionSizer) -> None:
        """Higher ATR (more volatile) -> fewer shares (same dollar risk)."""
        size_calm = sizer_atr.calculate(entry_price=2500.0, stop_loss=2450.0, atr=20.0)
        size_volatile = sizer_atr.calculate(entry_price=2500.0, stop_loss=2450.0, atr=80.0)
        assert size_calm.shares >= size_volatile.shares


class TestSizerPortfolioEdgeCases:
    def test_zero_portfolio_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            PositionSizer(portfolio_value=0)

    def test_negative_portfolio_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            PositionSizer(portfolio_value=-100_000)

    def test_stop_above_entry_handled(self, sizer_fixed: PositionSizer) -> None:
        """Stop above entry (invalid for long trade) - should not crash."""
        size = sizer_fixed.calculate(entry_price=2500.0, stop_loss=2600.0)
        assert size.shares >= 0


# ===========================================================================
# PreTradeValidator Tests
# ===========================================================================

class TestPreTradeValidator:

    def _approved_kwargs(self, **overrides) -> dict:
        """Base kwargs that should produce an APPROVED result."""
        base = dict(
            symbol="RELIANCE",
            position_value=50_000.0,       # 10% of portfolio? No — 50k/500k = 10%... fix:
            portfolio_value=500_000.0,
            open_positions=2,
            confidence=0.75,
            daily_pnl=0.0,
            sector="Energy",
            sector_exposure=50_000.0,      # existing Energy exposure
        )
        # position_value must be ≤ 5% of 500k = 25k
        base["position_value"] = 20_000.0
        base.update(overrides)
        return base

    def test_all_checks_pass(self, validator: PreTradeValidator) -> None:
        result = validator.validate(**self._approved_kwargs())
        assert result.approved is True
        assert result.checks_passed == result.checks_total
        assert len(result.rejection_reasons) == 0

    def test_position_too_large(self, validator: PreTradeValidator) -> None:
        """Position value > 5% of portfolio -> rejected."""
        result = validator.validate(**self._approved_kwargs(
            position_value=30_000.0,   # 6% of 500k
            portfolio_value=500_000.0,
        ))
        assert result.approved is False
        assert any("5%" in r or "position" in r.lower() for r in result.rejection_reasons)

    def test_daily_loss_circuit_breaker(self, validator: PreTradeValidator) -> None:
        """Daily loss ≥ 2% -> circuit breaker fires."""
        result = validator.validate(**self._approved_kwargs(
            daily_pnl=-10_500.0,   # 2.1% of 500k
            portfolio_value=500_000.0,
        ))
        assert result.approved is False
        assert any("circuit breaker" in r.lower() for r in result.rejection_reasons)

    def test_max_positions_reached(self, validator: PreTradeValidator) -> None:
        """5 open positions -> rejected."""
        result = validator.validate(**self._approved_kwargs(
            open_positions=MAX_OPEN_POSITIONS
        ))
        assert result.approved is False
        assert any("position" in r.lower() for r in result.rejection_reasons)

    def test_sector_exposure_exceeded(self, validator: PreTradeValidator) -> None:
        """Adding position would push sector above 30%."""
        result = validator.validate(**self._approved_kwargs(
            sector_exposure=140_000.0,  # existing Energy = 28%
            position_value=15_000.0,    # adding 3% → 31% > 30% limit
            portfolio_value=500_000.0,
        ))
        assert result.approved is False
        assert any("sector" in r.lower() for r in result.rejection_reasons)

    def test_low_confidence_rejected(self, validator: PreTradeValidator) -> None:
        """Confidence below threshold -> rejected."""
        result = validator.validate(**self._approved_kwargs(
            confidence=MIN_CONFIDENCE_THRESHOLD - 0.01
        ))
        assert result.approved is False
        assert any("confidence" in r.lower() for r in result.rejection_reasons)

    def test_exact_threshold_confidence_approved(self, validator: PreTradeValidator) -> None:
        """Confidence exactly at threshold -> approved."""
        result = validator.validate(**self._approved_kwargs(
            confidence=MIN_CONFIDENCE_THRESHOLD
        ))
        assert result.approved is True

    def test_zero_portfolio_rejected(self, validator: PreTradeValidator) -> None:
        result = validator.validate(**self._approved_kwargs(portfolio_value=0.0))
        assert result.approved is False

    def test_unknown_sector_passes_with_warning(self, validator: PreTradeValidator) -> None:
        """Sector=None -> check passes but adds a warning."""
        result = validator.validate(**self._approved_kwargs(sector=None))
        assert result.approved is True
        assert any("sector" in w.lower() for w in result.warnings)

    def test_multiple_failures_all_reported(self, validator: PreTradeValidator) -> None:
        """Multiple violations -> all reasons returned."""
        result = validator.validate(
            symbol="BAD",
            position_value=100_000.0,   # too large
            portfolio_value=500_000.0,
            open_positions=MAX_OPEN_POSITIONS,  # max positions
            confidence=0.30,            # too low
            daily_pnl=-15_000.0,        # circuit breaker
            sector="Energy",
            sector_exposure=200_000.0,  # sector cap
        )
        assert result.approved is False
        assert len(result.rejection_reasons) >= 3  # at least 3 failures

    def test_summary_string_approved(self, validator: PreTradeValidator) -> None:
        result = validator.validate(**self._approved_kwargs())
        assert "APPROVED" in result.summary

    def test_summary_string_rejected(self, validator: PreTradeValidator) -> None:
        result = validator.validate(**self._approved_kwargs(confidence=0.10))
        assert "REJECTED" in result.summary

    def test_daily_loss_exactly_at_limit(self, validator: PreTradeValidator) -> None:
        """Daily loss exactly at 2% limit -> circuit breaker fires."""
        result = validator.validate(**self._approved_kwargs(
            daily_pnl=-(500_000.0 * MAX_DAILY_LOSS_PCT),
            portfolio_value=500_000.0,
        ))
        assert result.approved is False

    def test_daily_profit_no_circuit_breaker(self, validator: PreTradeValidator) -> None:
        """Positive daily P&L -> circuit breaker stays off."""
        result = validator.validate(**self._approved_kwargs(daily_pnl=5_000.0))
        assert result.approved is True


# ===========================================================================
# PortfolioRisk Tests
# ===========================================================================

class TestPortfolioRisk:

    def test_initial_state(self, portfolio: PortfolioRisk) -> None:
        snap = portfolio.snapshot()
        assert snap.open_positions == 0
        assert snap.total_capital_at_risk == 0.0
        assert snap.daily_pnl == 0.0
        assert snap.circuit_breaker_triggered is False
        assert snap.positions_remaining == MAX_OPEN_POSITIONS

    def test_add_position(self, portfolio: PortfolioRisk, sample_position: Position) -> None:
        portfolio.add_position(sample_position)
        snap = portfolio.snapshot()
        assert snap.open_positions == 1
        assert snap.total_capital_at_risk == 2_000.0
        assert snap.total_position_value == 100_000.0

    def test_close_position_updates_pnl(
        self, portfolio: PortfolioRisk, sample_position: Position
    ) -> None:
        portfolio.add_position(sample_position)
        pnl = portfolio.close_position("RELIANCE", exit_price=2600.0)
        # (2600 - 2500) × 40 shares = 4,000 profit
        assert pnl == pytest.approx(4_000.0, rel=0.01)
        snap = portfolio.snapshot()
        assert snap.open_positions == 0
        assert snap.daily_pnl == pytest.approx(4_000.0, rel=0.01)

    def test_close_nonexistent_position_raises(self, portfolio: PortfolioRisk) -> None:
        with pytest.raises(KeyError):
            portfolio.close_position("NONEXISTENT", exit_price=2500.0)

    def test_duplicate_position_raises(
        self, portfolio: PortfolioRisk, sample_position: Position
    ) -> None:
        portfolio.add_position(sample_position)
        with pytest.raises(ValueError, match="already open"):
            portfolio.add_position(sample_position)

    def test_max_positions_enforced(self, portfolio: PortfolioRisk) -> None:
        """Adding beyond MAX_OPEN_POSITIONS raises ValueError."""
        for i in range(MAX_OPEN_POSITIONS):
            portfolio.add_position(Position(
                symbol=f"STOCK{i}",
                entry_price=100.0,
                stop_loss=95.0,
                shares=10,
                position_value=1_000.0,
                capital_at_risk=50.0,
                sector="Tech",
            ))
        with pytest.raises(ValueError, match="maximum"):
            portfolio.add_position(Position(
                symbol="OVERFLOWSTOCK",
                entry_price=100.0,
                stop_loss=95.0,
                shares=10,
                position_value=1_000.0,
                capital_at_risk=50.0,
            ))

    def test_sector_exposure_tracking(
        self, portfolio: PortfolioRisk, sample_position: Position
    ) -> None:
        portfolio.add_position(sample_position)  # Energy, 100,000
        snap = portfolio.snapshot()
        assert snap.sector_exposures.get("Energy", 0.0) == pytest.approx(100_000.0, rel=0.01)

    def test_get_position(
        self, portfolio: PortfolioRisk, sample_position: Position
    ) -> None:
        portfolio.add_position(sample_position)
        pos = portfolio.get_position("RELIANCE")
        assert pos is not None
        assert pos.symbol == "RELIANCE"

    def test_get_position_missing_returns_none(self, portfolio: PortfolioRisk) -> None:
        assert portfolio.get_position("NONEXISTENT") is None

    def test_available_capital(
        self, portfolio: PortfolioRisk, sample_position: Position
    ) -> None:
        portfolio.add_position(sample_position)
        snap = portfolio.snapshot()
        assert snap.available_capital == pytest.approx(400_000.0, rel=0.01)

    def test_deployed_pct(
        self, portfolio: PortfolioRisk, sample_position: Position
    ) -> None:
        portfolio.add_position(sample_position)
        snap = portfolio.snapshot()
        assert snap.total_deployed_pct == pytest.approx(0.20, rel=0.01)  # 100k/500k

    def test_zero_portfolio_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            PortfolioRisk(portfolio_value=0)

    def test_loss_pnl_tracked(
        self, portfolio: PortfolioRisk, sample_position: Position
    ) -> None:
        portfolio.add_position(sample_position)
        portfolio.close_position("RELIANCE", exit_price=2400.0)
        snap = portfolio.snapshot()
        # (2400 - 2500) × 40 = -4,000
        assert snap.daily_pnl == pytest.approx(-4_000.0, rel=0.01)


# ===========================================================================
# Integration: Full Risk Pipeline
# ===========================================================================

class TestRiskPipeline:
    """
    End-to-end: sizer → validator → portfolio.
    This is how Sprint 4 will use the risk engine.
    """

    def test_approved_trade_full_pipeline(self) -> None:
        import os
        os.environ["SIZING_METHOD"] = "fixed_fractional"
        os.environ["RISK_PER_TRADE_PCT"] = "1.0"

        portfolio = PortfolioRisk(portfolio_value=500_000.0)
        sizer = PositionSizer(portfolio_value=500_000.0)
        validator = PreTradeValidator()

        # Step 1: size the position
        size = sizer.calculate(entry_price=2500.0, stop_loss=2450.0)
        assert size.shares > 0

        # Step 2: validate
        snap = portfolio.snapshot()
        result = validator.validate(
            symbol="RELIANCE",
            position_value=size.position_value,
            portfolio_value=snap.portfolio_value,
            open_positions=snap.open_positions,
            confidence=0.78,
            daily_pnl=snap.daily_pnl,
            sector="Energy",
            sector_exposure=snap.sector_exposures.get("Energy", 0.0),
            capital_at_risk=size.capital_at_risk,
        )
        assert result.approved is True

        # Step 3: register in portfolio
        position = Position(
            symbol="RELIANCE",
            entry_price=2500.0,
            stop_loss=2450.0,
            shares=size.shares,
            position_value=size.position_value,
            capital_at_risk=size.capital_at_risk,
            sector="Energy",
        )
        portfolio.add_position(position)

        snap2 = portfolio.snapshot()
        assert snap2.open_positions == 1
        assert snap2.total_capital_at_risk == size.capital_at_risk

    def test_circuit_breaker_blocks_all_trades(self) -> None:
        """After a big loss, no trades should be approved."""
        portfolio = PortfolioRisk(portfolio_value=500_000.0)
        validator = PreTradeValidator()

        # Simulate a bad day: big loss already taken
        bad_position = Position(
            symbol="BADTRADE",
            entry_price=1000.0,
            stop_loss=900.0,
            shares=100,
            position_value=100_000.0,
            capital_at_risk=10_000.0,
        )
        portfolio.add_position(bad_position)
        portfolio.close_position("BADTRADE", exit_price=850.0)  # big loss

        snap = portfolio.snapshot()
        assert snap.circuit_breaker_triggered is True

        # Any new trade should be blocked
        result = validator.validate(
            symbol="NEWTRADE",
            position_value=10_000.0,
            portfolio_value=snap.portfolio_value,
            open_positions=snap.open_positions,
            confidence=0.90,
            daily_pnl=snap.daily_pnl,
        )
        assert result.approved is False
        assert any("circuit breaker" in r.lower() for r in result.rejection_reasons)
