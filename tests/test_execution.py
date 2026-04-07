"""
Unit tests for Sprint 4 execution layer.

Design:
- No external deps (no yfinance calls, no PostgreSQL)
- PaperBroker._fetch_live_price is patched with unittest.mock
- OrderManager uses ':memory:' SQLite so tests run in isolation
- PortfolioRisk is a real instance (it has no external deps)

Split:
- TestOrder / TestRiskDecision — pure dataclass contracts
- TestPaperBroker              — fill logic, slippage, close logic
- TestOrderManagerIdempotency  — the critical idempotency guarantees
- TestOrderManagerLifecycle    — submit → fill → close round-trip
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from src.execution.broker import Order, OrderSide, OrderStatus, RiskDecision
from src.execution.order_manager import OrderManager
from src.execution.paper_broker import PaperBroker
from src.risk_management.portfolio import PortfolioRisk, Position

# ============================================================
# Fixtures
# ============================================================


def _make_decision(
    symbol: str = "RELIANCE.NS",
    shares: int = 10,
    entry_price: float = 2800.0,
    stop_loss: float = 2720.0,
    take_profit: float = 2960.0,
    confidence: float = 0.75,
    capital_at_risk: float = 800.0,
) -> RiskDecision:
    return RiskDecision(
        symbol=symbol,
        side=OrderSide.BUY,
        shares=shares,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=confidence,
        strategy="multi_agent",
        agent_reasoning="Strong RSI + breakout confirmation",
        capital_at_risk=capital_at_risk,
    )


def _make_portfolio() -> PortfolioRisk:
    return PortfolioRisk(portfolio_value=100_000.0)


def _make_broker(live_price: float = 2805.0) -> tuple[PaperBroker, MagicMock]:
    """Returns (broker, mock_fetch) so tests can override the price."""
    portfolio = _make_portfolio()
    broker = PaperBroker(portfolio=portfolio, slippage_bps=5)
    mock_fetch = MagicMock(return_value=live_price)
    broker._fetch_live_price = mock_fetch  # type: ignore[method-assign]
    return broker, mock_fetch


def _make_manager(live_price: float = 2805.0) -> tuple[OrderManager, MagicMock]:
    broker, mock_fetch = _make_broker(live_price)
    manager = OrderManager(broker=broker, db_path=":memory:")
    return manager, mock_fetch


# ============================================================
# TestOrder — dataclass contracts
# ============================================================


class TestOrder:
    def test_from_risk_decision_maps_all_fields(self):
        decision = _make_decision()
        order = Order.from_risk_decision(decision)

        assert order.symbol == "RELIANCE.NS"
        assert order.side == OrderSide.BUY
        assert order.shares == 10
        assert order.requested_price == 2800.0
        assert order.stop_loss == 2720.0
        assert order.take_profit == 2960.0
        assert order.confidence == 0.75
        assert order.capital_at_risk == 800.0
        assert order.status == OrderStatus.PENDING

    def test_from_risk_decision_accepts_preset_order_id(self):
        decision = _make_decision()
        preset_id = "fixed-uuid-for-testing"
        order = Order.from_risk_decision(decision, order_id=preset_id)
        assert order.order_id == preset_id

    def test_from_risk_decision_generates_uuid_when_none(self):
        decision = _make_decision()
        order = Order.from_risk_decision(decision)
        # Should be a valid UUID string
        uuid.UUID(order.order_id)  # raises ValueError if invalid

    def test_position_value_is_zero_before_fill(self):
        order = Order.from_risk_decision(_make_decision())
        assert order.position_value == 0.0

    def test_position_value_after_fill(self):
        order = Order.from_risk_decision(_make_decision(shares=10))
        order.fill_price = 2805.14
        assert order.position_value == pytest.approx(2805.14 * 10)

    def test_is_open_only_when_filled(self):
        order = Order.from_risk_decision(_make_decision())
        assert not order.is_open
        order.status = OrderStatus.FILLED
        assert order.is_open

    def test_is_terminal_states(self):
        order = Order.from_risk_decision(_make_decision())
        for terminal_status in (
            OrderStatus.CLOSED,
            OrderStatus.STOPPED_OUT,
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
        ):
            order.status = terminal_status
            assert order.is_terminal
        order.status = OrderStatus.FILLED
        assert not order.is_terminal


# ============================================================
# TestPaperBroker — fill and close logic
# ============================================================


class TestPaperBroker:
    def test_submit_fills_at_live_price_with_slippage(self):
        broker, _ = _make_broker(live_price=2800.0)
        decision = _make_decision(entry_price=2790.0)
        order = Order.from_risk_decision(decision)

        result = broker.submit(order)

        assert result.status == OrderStatus.FILLED
        # BUY slippage = 2800 × 1.0005 = 2801.40
        assert result.fill_price == pytest.approx(2800.0 * 1.0005, rel=1e-4)
        assert result.fill_timestamp is not None

    def test_submit_calculates_slippage_relative_to_requested_price(self):
        broker, _ = _make_broker(live_price=2800.0)
        decision = _make_decision(entry_price=2790.0)
        order = Order.from_risk_decision(decision)

        result = broker.submit(order)

        # slippage = fill_price - requested_price
        assert result.slippage == pytest.approx(result.fill_price - 2790.0, rel=1e-4)

    def test_submit_rejects_when_price_unavailable(self):
        portfolio = _make_portfolio()
        broker = PaperBroker(portfolio=portfolio)
        broker._fetch_live_price = MagicMock(return_value=None)  # type: ignore[method-assign]

        order = Order.from_risk_decision(_make_decision())
        result = broker.submit(order)

        assert result.status == OrderStatus.REJECTED
        assert result.fill_price is None

    def test_submit_ignores_non_pending_order(self):
        broker, mock_fetch = _make_broker(live_price=2800.0)
        order = Order.from_risk_decision(_make_decision())
        order.status = OrderStatus.FILLED  # already filled

        result = broker.submit(order)

        mock_fetch.assert_not_called()
        assert result.status == OrderStatus.FILLED  # unchanged

    def test_close_position_calculates_pnl(self):
        # Use submit() so PortfolioRisk has a record of the position before we close it.
        broker, _ = _make_broker(live_price=2900.0)
        decision = _make_decision(shares=10, capital_at_risk=800.0)
        order = Order.from_risk_decision(decision)
        # Simulate a prior fill by registering the position directly in the portfolio.
        broker._portfolio.add_position(
            Position(
                symbol=order.symbol,
                entry_price=2800.0,
                stop_loss=order.stop_loss,
                shares=order.shares,
                position_value=2800.0 * order.shares,
                capital_at_risk=order.capital_at_risk,
            )
        )
        order.status = OrderStatus.FILLED
        order.fill_price = 2800.0

        result = broker.close_position(order, reason="take_profit")

        assert result.status == OrderStatus.CLOSED
        assert result.exit_reason == "take_profit"
        # sell with 5 bps slippage: 2900 × 0.9995 = 2898.55
        expected_exit = round(2900.0 * 0.9995, 2)
        assert result.exit_price == pytest.approx(expected_exit, rel=1e-4)
        # pnl = (exit - entry) × shares
        expected_pnl = (expected_exit - 2800.0) * 10
        assert result.pnl == pytest.approx(expected_pnl, rel=1e-3)

    def test_close_position_sets_stopped_out_status(self):
        broker, _ = _make_broker(live_price=2720.0)
        decision = _make_decision(shares=5)
        order = Order.from_risk_decision(decision)
        broker._portfolio.add_position(
            Position(
                symbol=order.symbol,
                entry_price=2800.0,
                stop_loss=order.stop_loss,
                shares=order.shares,
                position_value=2800.0 * order.shares,
                capital_at_risk=order.capital_at_risk,
            )
        )
        order.status = OrderStatus.FILLED
        order.fill_price = 2800.0

        result = broker.close_position(order, reason="stop_loss")

        assert result.status == OrderStatus.STOPPED_OUT

    def test_close_position_calculates_r_multiple(self):
        broker, _ = _make_broker(live_price=2900.0)
        decision = _make_decision(shares=10, capital_at_risk=800.0)
        order = Order.from_risk_decision(decision)
        broker._portfolio.add_position(
            Position(
                symbol=order.symbol,
                entry_price=2800.0,
                stop_loss=order.stop_loss,
                shares=order.shares,
                position_value=2800.0 * order.shares,
                capital_at_risk=order.capital_at_risk,
            )
        )
        order.status = OrderStatus.FILLED
        order.fill_price = 2800.0

        result = broker.close_position(order, reason="take_profit")

        # r_multiple = round(pnl / capital_at_risk, 2) — match the rounding in paper_broker
        assert result.r_multiple is not None
        assert result.r_multiple == round(result.pnl / 800.0, 2)  # type: ignore[operator]

    def test_close_ignores_non_filled_order(self):
        broker, mock_fetch = _make_broker(live_price=2900.0)
        order = Order.from_risk_decision(_make_decision())
        order.status = OrderStatus.PENDING

        result = broker.close_position(order, reason="manual")

        mock_fetch.assert_not_called()
        assert result.status == OrderStatus.PENDING  # unchanged


# ============================================================
# TestOrderManagerIdempotency — the critical guarantee
# ============================================================


class TestOrderManagerIdempotency:
    def test_same_order_id_returns_existing_order(self):
        manager, _ = _make_manager(live_price=2805.0)
        decision = _make_decision()
        preset_id = "idempotency-test-id"

        first = manager.submit(decision, order_id=preset_id)
        second = manager.submit(decision, order_id=preset_id)

        assert first.order_id == second.order_id
        assert first.fill_price == second.fill_price
        assert first.fill_timestamp == second.fill_timestamp

    def test_broker_called_only_once_for_same_order_id(self):
        broker, mock_fetch = _make_broker(live_price=2805.0)
        manager = OrderManager(broker=broker, db_path=":memory:")
        decision = _make_decision()
        preset_id = "broker-call-count-test"

        manager.submit(decision, order_id=preset_id)
        manager.submit(decision, order_id=preset_id)
        manager.submit(decision, order_id=preset_id)

        # _fetch_live_price should have been called exactly once
        assert mock_fetch.call_count == 1

    def test_different_order_ids_create_separate_orders(self):
        manager, _ = _make_manager(live_price=2805.0)

        # Use different symbols — PortfolioRisk rejects two open positions for the same symbol.
        order_a = manager.submit(_make_decision(symbol="RELIANCE.NS"), order_id="id-a")
        order_b = manager.submit(_make_decision(symbol="TCS.NS"), order_id="id-b")

        assert order_a.order_id != order_b.order_id
        assert len(manager.get_all_orders()) == 2


# ============================================================
# TestOrderManagerLifecycle — submit → open → close round-trip
# ============================================================


class TestOrderManagerLifecycle:
    def test_submit_stores_pending_then_filled(self):
        manager, _ = _make_manager(live_price=2805.0)
        decision = _make_decision()

        order = manager.submit(decision)

        assert order.status == OrderStatus.FILLED
        # Verify it's persisted
        loaded = manager._load_order(order.order_id)
        assert loaded is not None
        assert loaded.status == OrderStatus.FILLED
        assert loaded.fill_price is not None

    def test_get_open_positions_returns_filled_orders(self):
        manager, _ = _make_manager(live_price=2805.0)

        manager.submit(_make_decision(symbol="RELIANCE.NS"), order_id="pos-1")
        manager.submit(_make_decision(symbol="TCS.NS"), order_id="pos-2")

        open_positions = manager.get_open_positions()
        assert len(open_positions) == 2

    def test_close_position_removes_from_open_positions(self):
        manager, _ = _make_manager(live_price=2805.0)
        order = manager.submit(_make_decision(), order_id="to-close")

        manager.close_position(order.order_id, reason="manual")

        open_positions = manager.get_open_positions()
        assert len(open_positions) == 0

    def test_close_nonexistent_order_returns_none(self):
        manager, _ = _make_manager()
        result = manager.close_position("nonexistent-id")
        assert result is None

    def test_close_already_closed_order_returns_unchanged(self):
        manager, _ = _make_manager(live_price=2805.0)
        order = manager.submit(_make_decision(), order_id="already-closed")
        manager.close_position(order.order_id, reason="manual")  # close once

        result = manager.close_position(order.order_id, reason="manual")  # try again

        assert result is not None
        assert result.status == OrderStatus.CLOSED  # unchanged

    def test_rejected_order_is_persisted(self):
        broker, _ = _make_broker(live_price=2805.0)
        broker._fetch_live_price = MagicMock(return_value=None)  # type: ignore[method-assign]
        manager = OrderManager(broker=broker, db_path=":memory:")

        order = manager.submit(_make_decision())

        assert order.status == OrderStatus.REJECTED
        loaded = manager._load_order(order.order_id)
        assert loaded is not None
        assert loaded.status == OrderStatus.REJECTED

    def test_get_all_orders_includes_all_statuses(self):
        manager, _ = _make_manager(live_price=2805.0)

        o1 = manager.submit(_make_decision(symbol="RELIANCE.NS"), order_id="all-1")
        o2 = manager.submit(_make_decision(symbol="TCS.NS"), order_id="all-2")
        manager.close_position(o1.order_id, reason="manual")

        all_orders = manager.get_all_orders()
        assert len(all_orders) == 2
        statuses = {o.status for o in all_orders}
        assert OrderStatus.CLOSED in statuses
        assert OrderStatus.FILLED in statuses
