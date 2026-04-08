"""
Unit tests for the Sprint 5 analytics layer.

Design principle: all tests use synthetic ClosedTrade fixtures with known
values. No database, no external calls, no mocking required.

Verify the metrics manually before running:
  - 3 trades: +₹1000, +₹500, -₹300 → total P&L = ₹1200
  - Win rate = 2/3 ≈ 0.667
  - Gross profit = 1500, gross loss = 300 → profit factor = 5.0
  - Expectancy = (0.667 × 750) - (0.333 × 300) = 500 - 99.9 = 400.1
"""

from datetime import datetime, timedelta

import pytest

from src.journal.analytics import PerformanceEngine, _aggregate_daily_pnl, _std
from src.journal.models import ClosedTrade, _parse_timestamp

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_DATE = datetime(2024, 3, 1, 10, 0, 0)


def _make_trade(
    order_id: str,
    pnl: float,
    days_after_base: int = 0,
    hold_days: int = 1,
    r_multiple: float = 1.0,
    stop_loss: float = 95.0,
    entry_price: float = 100.0,
    strategy: str = "momentum",
    symbol: str = "RELIANCE",
) -> ClosedTrade:
    """Helper to create ClosedTrade fixtures without boilerplate."""
    entry_ts = BASE_DATE + timedelta(days=days_after_base)
    exit_ts = entry_ts + timedelta(days=hold_days)
    return ClosedTrade(
        order_id=order_id,
        symbol=symbol,
        strategy=strategy,
        entry_timestamp=entry_ts,
        exit_timestamp=exit_ts,
        entry_price=entry_price,
        exit_price=entry_price + (pnl / 10),  # 10 shares assumption for simplicity
        shares=10,
        stop_loss=stop_loss,
        status="CLOSED" if pnl >= 0 else "STOPPED_OUT",
        pnl=pnl,
        pnl_pct=round(pnl / (entry_price * 10) * 100, 2),
        hold_days=hold_days,
        r_multiple=r_multiple,
    )


# Three-trade base set: two winners, one loser
TRADE_W1 = _make_trade("T001", pnl=1000.0, days_after_base=0, r_multiple=2.0)
TRADE_W2 = _make_trade("T002", pnl=500.0, days_after_base=1, r_multiple=1.0)
TRADE_L1 = _make_trade("T003", pnl=-300.0, days_after_base=2, r_multiple=-0.6)

THREE_TRADES = [TRADE_W1, TRADE_W2, TRADE_L1]


# ---------------------------------------------------------------------------
# ClosedTrade.from_db_row tests
# ---------------------------------------------------------------------------


class TestClosedTradeFromDbRow:
    def test_basic_construction(self):
        row = {
            "order_id": "X001",
            "symbol": "TCS",
            "strategy": "breakout",
            "fill_timestamp": "2024-03-01 10:00:00",
            "exit_timestamp": "2024-03-05 15:30:00",
            "entry_price": 3500.0,
            "exit_price": 3600.0,
            "shares": 5,
            "stop_loss": 3450.0,
            "status": "CLOSED",
            "pnl": 500.0,
            "pnl_pct": 2.86,
        }
        trade = ClosedTrade.from_db_row(row)
        assert trade.order_id == "X001"
        assert trade.symbol == "TCS"
        assert trade.hold_days == 4
        # r_multiple = 500 / ((3500 - 3450) * 5) = 500 / 250 = 2.0
        assert trade.r_multiple == pytest.approx(2.0, rel=1e-3)

    def test_hold_days_same_day(self):
        row = {
            "order_id": "X002",
            "symbol": "INFY",
            "strategy": "technical",
            "fill_timestamp": "2024-03-01 10:00:00",
            "exit_timestamp": "2024-03-01 15:00:00",
            "entry_price": 1500.0,
            "exit_price": 1480.0,
            "shares": 10,
            "stop_loss": 1470.0,
            "status": "STOPPED_OUT",
            "pnl": -200.0,
            "pnl_pct": -1.33,
        }
        trade = ClosedTrade.from_db_row(row)
        assert trade.hold_days == 0

    def test_zero_stop_loss_risk_gives_zero_r(self):
        row = {
            "order_id": "X003",
            "symbol": "HDFC",
            "strategy": "momentum",
            "fill_timestamp": "2024-03-01 10:00:00",
            "exit_timestamp": "2024-03-02 10:00:00",
            "entry_price": 1000.0,
            "exit_price": 1050.0,
            "shares": 5,
            "stop_loss": 1000.0,  # stop == entry → zero initial risk
            "status": "CLOSED",
            "pnl": 250.0,
            "pnl_pct": 5.0,
        }
        trade = ClosedTrade.from_db_row(row)
        assert trade.r_multiple == 0.0


# ---------------------------------------------------------------------------
# _parse_timestamp tests
# ---------------------------------------------------------------------------


class TestParseTimestamp:
    def test_datetime_passthrough(self):
        dt = datetime(2024, 3, 1, 9, 15)
        assert _parse_timestamp(dt) == dt

    def test_iso_string_no_microseconds(self):
        result = _parse_timestamp("2024-03-01 09:15:30")
        assert result == datetime(2024, 3, 1, 9, 15, 30)

    def test_iso_string_with_microseconds(self):
        result = _parse_timestamp("2024-03-01 09:15:30.123456")
        assert result == datetime(2024, 3, 1, 9, 15, 30, 123456)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_timestamp("not-a-date")


# ---------------------------------------------------------------------------
# PerformanceEngine tests — verified against hand-calculated values
# ---------------------------------------------------------------------------


class TestWinRate:
    def test_two_of_three(self):
        result = PerformanceEngine.win_rate(THREE_TRADES)
        assert result == pytest.approx(2 / 3, rel=1e-3)

    def test_all_winners(self):
        trades = [_make_trade(f"T{i}", pnl=100.0) for i in range(5)]
        assert PerformanceEngine.win_rate(trades) == pytest.approx(1.0)

    def test_all_losers(self):
        trades = [_make_trade(f"T{i}", pnl=-100.0) for i in range(3)]
        assert PerformanceEngine.win_rate(trades) == pytest.approx(0.0)

    def test_empty_returns_none(self):
        assert PerformanceEngine.win_rate([]) is None


class TestProfitFactor:
    def test_three_trades(self):
        # gross_profit=1500, gross_loss=300 → pf=5.0
        result = PerformanceEngine.profit_factor(THREE_TRADES)
        assert result == pytest.approx(5.0, rel=1e-3)

    def test_no_losers_returns_none(self):
        trades = [_make_trade(f"T{i}", pnl=100.0) for i in range(3)]
        assert PerformanceEngine.profit_factor(trades) is None

    def test_empty_returns_none(self):
        assert PerformanceEngine.profit_factor([]) is None


class TestExpectancy:
    def test_three_trades(self):
        # winners: [1000, 500] → avg_win=750; losers: [-300] → avg_loss=300
        # win_rate=2/3, loss_rate=1/3
        # expectancy = (2/3 × 750) - (1/3 × 300) = 500 - 100 = 400
        result = PerformanceEngine.expectancy(THREE_TRADES)
        assert result == pytest.approx(400.0, rel=1e-2)

    def test_empty_returns_none(self):
        assert PerformanceEngine.expectancy([]) is None

    def test_all_losers_positive_loss_rate(self):
        trades = [_make_trade(f"T{i}", pnl=-200.0) for i in range(3)]
        result = PerformanceEngine.expectancy(trades)
        # win_rate=0, avg_win=0, loss_rate=1, avg_loss=200 → -200
        assert result == pytest.approx(-200.0, rel=1e-2)


class TestAvgRMultiple:
    def test_three_trades(self):
        # r_multiples: 2.0, 1.0, -0.6 → avg = 0.8
        result = PerformanceEngine.avg_r_multiple(THREE_TRADES)
        assert result == pytest.approx(0.8, rel=1e-3)

    def test_empty_returns_none(self):
        assert PerformanceEngine.avg_r_multiple([]) is None


class TestSharpeRatio:
    def test_single_trade_returns_none(self):
        trades = [_make_trade("T001", pnl=100.0)]
        assert PerformanceEngine.sharpe_ratio(trades) is None

    def test_empty_returns_none(self):
        assert PerformanceEngine.sharpe_ratio([]) is None

    def test_returns_float_with_multiple_days(self):
        # Create trades on 10 different days
        trades = [
            _make_trade(f"T{i:03d}", pnl=100.0 if i % 3 != 0 else -50.0, days_after_base=i)
            for i in range(10)
        ]
        result = PerformanceEngine.sharpe_ratio(trades)
        assert result is not None
        assert isinstance(result, float)

    def test_all_same_pnl_returns_none(self):
        # Zero standard deviation → Sharpe undefined
        trades = [_make_trade(f"T{i:03d}", pnl=100.0, days_after_base=i) for i in range(5)]
        result = PerformanceEngine.sharpe_ratio(trades)
        assert result is None


class TestMaxDrawdown:
    def test_single_trade_returns_none(self):
        assert PerformanceEngine.max_drawdown([_make_trade("T1", 100.0)]) is None

    def test_empty_returns_none(self):
        assert PerformanceEngine.max_drawdown([]) is None

    def test_monotonically_increasing_equity(self):
        # No drawdown if each trade is profitable
        trades = [_make_trade(f"T{i:03d}", pnl=200.0, days_after_base=i) for i in range(5)]
        assert PerformanceEngine.max_drawdown(trades) == 0.0

    def test_known_drawdown(self):
        # Equity curve: +1000, +500, -800, +200
        # Cumulative:   1000, 1500, 700, 900
        # Peak at 1500, trough at 700 → drawdown = -800
        trades = [
            _make_trade("T1", pnl=1000.0, days_after_base=0),
            _make_trade("T2", pnl=500.0, days_after_base=1),
            _make_trade("T3", pnl=-800.0, days_after_base=2),
            _make_trade("T4", pnl=200.0, days_after_base=3),
        ]
        result = PerformanceEngine.max_drawdown(trades)
        assert result == pytest.approx(-800.0, rel=1e-3)


class TestMaxDrawdownPct:
    def test_known_percentage(self):
        trades = [
            _make_trade("T1", pnl=1000.0, days_after_base=0),
            _make_trade("T2", pnl=-500.0, days_after_base=1),
        ]
        # Equity: 1000 → 500, peak=1000, drawdown=-500
        # As pct of ₹1,00,000 initial capital = -0.5%
        result = PerformanceEngine.max_drawdown_pct(trades, initial_capital=100_000)
        assert result == pytest.approx(-0.5, rel=1e-2)


class TestByStrategy:
    def test_groups_correctly(self):
        t1 = _make_trade("T1", pnl=500.0, strategy="momentum")
        t2 = _make_trade("T2", pnl=-200.0, strategy="momentum")
        t3 = _make_trade("T3", pnl=800.0, strategy="breakout")
        result = PerformanceEngine.by_strategy([t1, t2, t3])
        assert "momentum" in result
        assert "breakout" in result
        assert result["momentum"]["trades"] == 2
        assert result["breakout"]["trades"] == 1
        assert result["breakout"]["total_pnl"] == pytest.approx(800.0)

    def test_empty_returns_empty_dict(self):
        assert PerformanceEngine.by_strategy([]) == {}


class TestSummary:
    def test_three_trades_structure(self):
        result = PerformanceEngine.summary(THREE_TRADES)
        expected_keys = {
            "total_trades",
            "total_pnl",
            "win_rate",
            "profit_factor",
            "expectancy",
            "avg_r_multiple",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "avg_win",
            "avg_loss",
            "avg_hold_days",
            "winners",
            "losers",
        }
        assert expected_keys == set(result.keys())
        assert result["total_trades"] == 3
        assert result["total_pnl"] == pytest.approx(1200.0)
        assert result["winners"] == 2
        assert result["losers"] == 1

    def test_empty_trades(self):
        result = PerformanceEngine.summary([])
        assert result["total_trades"] == 0
        assert result["total_pnl"] == 0.0
        assert result["win_rate"] is None


# ---------------------------------------------------------------------------
# Private helper tests
# ---------------------------------------------------------------------------


class TestAggregateDaily:
    def test_same_day_trades_aggregated(self):
        t1 = _make_trade("T1", pnl=300.0, days_after_base=0, hold_days=0)
        t2 = _make_trade("T2", pnl=200.0, days_after_base=0, hold_days=0)
        result = _aggregate_daily_pnl([t1, t2])
        assert len(result) == 1
        assert list(result.values())[0] == pytest.approx(500.0)


class TestStd:
    def test_known_std(self):
        # [2, 4, 4, 4, 5, 5, 7, 9] → population std = 2.0
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        assert _std(values) == pytest.approx(2.0, rel=1e-3)

    def test_single_value_returns_zero(self):
        assert _std([5.0]) == 0.0
