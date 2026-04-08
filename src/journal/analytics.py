"""
Performance analytics engine.

Design constraints (enforced):
  - Zero database dependencies. No sqlite3, no psycopg2.
  - Pure functions: same inputs always produce same outputs.
  - All functions take list[ClosedTrade] and return a plain value or dict.
  - Edge cases (empty list, division by zero) are handled explicitly.

Why pure functions?
  Testing is trivial: pass synthetic trade data, assert on the output.
  No DB setup, no mocking, no fixtures beyond a list of dataclasses.
  The entire analytics layer can be verified in milliseconds.

Usage:
    from src.journal.analytics import PerformanceEngine
    trades = repo.get_closed_trades()
    metrics = PerformanceEngine.summary(trades)
"""

from __future__ import annotations

import math
from collections import defaultdict

from src.journal.models import ClosedTrade


class PerformanceEngine:
    """Stateless analytics over a list of ClosedTrade objects.

    All methods are @staticmethod — instantiation is never required.
    """

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    @staticmethod
    def win_rate(trades: list[ClosedTrade]) -> float | None:
        """Fraction of trades with positive P&L.

        Returns None if the trade list is empty (avoids misleading 0.0).

        Args:
            trades: List of closed trades.

        Returns:
            Float in [0, 1], or None if no trades.
        """
        if not trades:
            return None
        winners = sum(1 for t in trades if t.pnl > 0)
        return winners / len(trades)

    @staticmethod
    def profit_factor(trades: list[ClosedTrade]) -> float | None:
        """Gross profit / gross loss.

        A value above 1.0 means the strategy is profitable before commissions.
        Returns None if there are no losing trades (can't divide by zero)
        or if the trade list is empty.

        Args:
            trades: List of closed trades.

        Returns:
            Float > 0, or None if undefined.
        """
        if not trades:
            return None
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        if gross_loss == 0:
            return None  # No losses: undefined (not infinite — avoid misleading number)
        return round(gross_profit / gross_loss, 3)

    @staticmethod
    def expectancy(trades: list[ClosedTrade]) -> float | None:
        """Expected P&L per trade in rupees.

        Formula: (win_rate × avg_win) - (loss_rate × avg_loss)

        A positive expectancy means the strategy has a mathematical edge.

        Args:
            trades: List of closed trades.

        Returns:
            Float (rupees per trade), or None if no trades.
        """
        if not trades:
            return None
        winners = [t.pnl for t in trades if t.pnl > 0]
        losers = [t.pnl for t in trades if t.pnl < 0]

        win_rate = len(winners) / len(trades)
        loss_rate = 1 - win_rate

        avg_win = sum(winners) / len(winners) if winners else 0.0
        avg_loss = abs(sum(losers) / len(losers)) if losers else 0.0

        return round((win_rate * avg_win) - (loss_rate * avg_loss), 2)

    @staticmethod
    def avg_r_multiple(trades: list[ClosedTrade]) -> float | None:
        """Average R-multiple across all trades.

        R-multiple expresses P&L in units of initial risk (1R).
        An average R of 0.5 means: for every rupee risked, you made 0.5R back.
        A strategy with 40% win rate can still be profitable if avg R > 1.5.

        Args:
            trades: List of closed trades.

        Returns:
            Float, or None if no trades.
        """
        if not trades:
            return None
        return round(sum(t.r_multiple for t in trades) / len(trades), 3)

    @staticmethod
    def sharpe_ratio(trades: list[ClosedTrade], risk_free_rate: float = 0.065) -> float | None:
        """Annualised Sharpe ratio from daily P&L.

        Formula: (mean_daily_return - daily_risk_free) / std_daily_return × √252

        Uses 6.5% annual risk-free rate (approximate Indian T-bill rate) by default.
        Requires at least 2 trades with different exit dates.

        Args:
            trades: List of closed trades.
            risk_free_rate: Annual risk-free rate as a decimal (default 6.5%).

        Returns:
            Float, or None if insufficient data.
        """
        daily_pnl = _aggregate_daily_pnl(trades)
        if len(daily_pnl) < 2:
            return None

        returns = list(daily_pnl.values())
        mean_r = sum(returns) / len(returns)
        std_r = _std(returns)

        if std_r == 0:
            return None

        # Daily risk-free rate
        daily_rf = risk_free_rate / 252
        sharpe = (mean_r - daily_rf) / std_r * math.sqrt(252)
        return round(sharpe, 3)

    @staticmethod
    def sortino_ratio(trades: list[ClosedTrade], risk_free_rate: float = 0.065) -> float | None:
        """Annualised Sortino ratio — only penalises downside volatility.

        More appropriate than Sharpe for strategies with asymmetric returns
        (large wins, small losses) because it doesn't penalise upside moves.

        Args:
            trades: List of closed trades.
            risk_free_rate: Annual risk-free rate as a decimal.

        Returns:
            Float, or None if insufficient data.
        """
        daily_pnl = _aggregate_daily_pnl(trades)
        if len(daily_pnl) < 2:
            return None

        returns = list(daily_pnl.values())
        mean_r = sum(returns) / len(returns)
        daily_rf = risk_free_rate / 252

        downside = [r for r in returns if r < daily_rf]
        if not downside:
            return None  # No losing days — sortino is undefined

        downside_std = _std(downside)
        if downside_std == 0:
            return None

        sortino = (mean_r - daily_rf) / downside_std * math.sqrt(252)
        return round(sortino, 3)

    @staticmethod
    def max_drawdown(trades: list[ClosedTrade]) -> float | None:
        """Maximum peak-to-trough decline in the equity curve (in rupees).

        Calculated on the running cumulative P&L, not per-trade.
        Returns a negative number (e.g. -12500 means ₹12,500 drawdown).
        Returns None if there are fewer than 2 trades.

        Args:
            trades: List of closed trades, will be sorted by exit_timestamp.

        Returns:
            Float (negative), or None if insufficient data.
        """
        if len(trades) < 2:
            return None

        sorted_trades = sorted(trades, key=lambda t: t.exit_timestamp)
        equity = 0.0
        peak = 0.0
        max_dd = 0.0

        for trade in sorted_trades:
            equity += trade.pnl
            if equity > peak:
                peak = equity
            dd = equity - peak
            if dd < max_dd:
                max_dd = dd

        return round(max_dd, 2) if max_dd < 0 else 0.0

    @staticmethod
    def max_drawdown_pct(
        trades: list[ClosedTrade], initial_capital: float = 100_000
    ) -> float | None:
        """Max drawdown expressed as a percentage of initial capital.

        Args:
            trades: List of closed trades.
            initial_capital: Starting capital in rupees (default ₹1,00,000).

        Returns:
            Float as percentage (e.g. -12.5 means 12.5% drawdown), or None.
        """
        dd = PerformanceEngine.max_drawdown(trades)
        if dd is None or initial_capital <= 0:
            return None
        return round((dd / initial_capital) * 100, 2)

    # ------------------------------------------------------------------
    # Breakdown analytics
    # ------------------------------------------------------------------

    @staticmethod
    def by_strategy(trades: list[ClosedTrade]) -> dict[str, dict]:
        """Break down key metrics per strategy tag.

        Args:
            trades: List of closed trades.

        Returns:
            Dict keyed by strategy name, each value a metrics dict.
        """
        grouped: dict[str, list[ClosedTrade]] = defaultdict(list)
        for t in trades:
            grouped[t.strategy].append(t)

        result = {}
        for strategy, strat_trades in grouped.items():
            result[strategy] = {
                "trades": len(strat_trades),
                "win_rate": PerformanceEngine.win_rate(strat_trades),
                "profit_factor": PerformanceEngine.profit_factor(strat_trades),
                "avg_r_multiple": PerformanceEngine.avg_r_multiple(strat_trades),
                "total_pnl": round(sum(t.pnl for t in strat_trades), 2),
                "expectancy": PerformanceEngine.expectancy(strat_trades),
            }
        return result

    @staticmethod
    def summary(trades: list[ClosedTrade]) -> dict:
        """Compute all top-level metrics in a single call.

        This is the primary entry point for the analytics dashboard.
        All values are JSON-serialisable (float, int, or None).

        Args:
            trades: List of closed trades.

        Returns:
            Dict with all performance metrics.
        """
        total_pnl = round(sum(t.pnl for t in trades), 2) if trades else 0.0
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl < 0]

        avg_win = round(sum(t.pnl for t in winners) / len(winners), 2) if winners else None
        avg_loss = round(sum(t.pnl for t in losers) / len(losers), 2) if losers else None
        avg_hold = round(sum(t.hold_days for t in trades) / len(trades), 1) if trades else None

        return {
            "total_trades": len(trades),
            "total_pnl": total_pnl,
            "win_rate": PerformanceEngine.win_rate(trades),
            "profit_factor": PerformanceEngine.profit_factor(trades),
            "expectancy": PerformanceEngine.expectancy(trades),
            "avg_r_multiple": PerformanceEngine.avg_r_multiple(trades),
            "sharpe_ratio": PerformanceEngine.sharpe_ratio(trades),
            "sortino_ratio": PerformanceEngine.sortino_ratio(trades),
            "max_drawdown": PerformanceEngine.max_drawdown(trades),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_hold_days": avg_hold,
            "winners": len(winners),
            "losers": len(losers),
        }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _aggregate_daily_pnl(trades: list[ClosedTrade]) -> dict[str, float]:
    """Sum P&L by exit date (ISO string key)."""
    daily: dict[str, float] = defaultdict(float)
    for t in trades:
        day = t.exit_timestamp.date().isoformat()
        daily[day] += t.pnl
    return dict(sorted(daily.items()))


def _std(values: list[float]) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)
