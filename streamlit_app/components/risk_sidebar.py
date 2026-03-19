"""
Risk Sidebar Component — Sprint 3

Renders a live portfolio risk panel in the Streamlit sidebar.

Responsibilities:
  - Show portfolio-level metrics (deployed capital, total risk, daily P&L)
  - Show circuit breaker status (green / amber / red)
  - Show open position count vs max
  - Show sector exposure breakdown
  - Show position size calculator for a selected stock

Design: Pure rendering functions — no state mutation.
All data comes in as arguments; nothing is fetched here.
This makes the component testable and reusable across pages.

Usage in app.py:
    from streamlit_app.components.risk_sidebar import render_risk_sidebar

    with st.sidebar:
        render_risk_sidebar(
            snapshot=portfolio.snapshot(),
            sizing_method=os.getenv("SIZING_METHOD", "fixed_fractional"),
        )
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from src.risk_management.portfolio import PortfolioSnapshot

# Colour tokens — consistent with the rest of the dashboard
_GREEN = "#00C853"
_AMBER = "#FFB300"
_RED = "#D50000"
_MUTED = "#888888"


def render_risk_sidebar(
    snapshot: PortfolioSnapshot,
    sizing_method: str = "fixed_fractional",
    selected_symbol: Optional[str] = None,
    proposed_size: Optional[dict] = None,
) -> None:
    """
    Render the full risk panel in the Streamlit sidebar.

    Args:
        snapshot:        Current portfolio snapshot from PortfolioRisk.snapshot()
        sizing_method:   Active sizing method name (for display only)
        selected_symbol: If a stock is selected, show its proposed sizing
        proposed_size:   Dict with keys: shares, position_value, capital_at_risk,
                         fraction_used, reasoning, approved (bool), rejection_reasons (list)
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🛡️ Risk Dashboard")

    _render_circuit_breaker(snapshot)
    _render_portfolio_metrics(snapshot, sizing_method)
    _render_position_slots(snapshot)
    _render_sector_exposure(snapshot)

    if selected_symbol and proposed_size:
        _render_proposed_trade(selected_symbol, proposed_size)

    st.sidebar.markdown("---")
    st.sidebar.caption("Risk limits: 5% max position · 2% daily loss · 5 max positions · 30% sector cap")


# ---------------------------------------------------------------------------
# Private render helpers
# ---------------------------------------------------------------------------

def _render_circuit_breaker(snapshot: PortfolioSnapshot) -> None:
    """Top-of-sidebar status pill — the most important signal."""
    daily_loss_pct = snapshot.daily_loss_pct * 100
    daily_pnl = snapshot.daily_pnl

    if snapshot.circuit_breaker_triggered:
        st.sidebar.error(
            f"🚨 **CIRCUIT BREAKER ACTIVE**\n\n"
            f"Daily loss: {abs(daily_pnl):,.0f} ({daily_loss_pct:.1f}%)\n\n"
            f"No new trades until tomorrow."
        )
        return

    # Traffic-light colouring based on how close we are to the 2% limit
    # 0–75% of limit → green, 75–90% → amber, 90–100% → red
    limit_used_fraction = daily_loss_pct / 2.0  # 2.0% is the limit

    if daily_pnl >= 0:
        pnl_str = f"+{daily_pnl:,.0f}"
        st.sidebar.success(f"✅ **Trading Active** · P&L today: {pnl_str}")
    elif limit_used_fraction < 0.75:
        st.sidebar.success(
            f"✅ **Trading Active** · Loss today: {abs(daily_pnl):,.0f} "
            f"({daily_loss_pct:.1f}%)"
        )
    elif limit_used_fraction < 0.90:
        st.sidebar.warning(
            f"⚠️ **Approaching Daily Limit** · Loss: {abs(daily_pnl):,.0f} "
            f"({daily_loss_pct:.1f}% of 2% limit)"
        )
    else:
        st.sidebar.error(
            f"🔴 **Near Circuit Breaker** · Loss: {abs(daily_pnl):,.0f} "
            f"({daily_loss_pct:.1f}% of 2% limit)"
        )


def _render_portfolio_metrics(snapshot: PortfolioSnapshot, sizing_method: str) -> None:
    """Capital deployment and risk metrics."""
    st.sidebar.markdown("### 📊 Portfolio Metrics")

    deployed_pct = snapshot.total_deployed_pct * 100
    risk_pct = snapshot.total_risk_pct * 100
    available = snapshot.available_capital

    col1, col2 = st.sidebar.columns(2)

    with col1:
        st.metric(
            label="Capital Deployed",
            value=f"{deployed_pct:.1f}%",
            delta=None,
        )
        st.metric(
            label="Available",
            value=f"{available:,.0f}",
        )

    with col2:
        # Risk % uses colour context: low=good, high=warning
        risk_color = _GREEN if risk_pct < 3 else (_AMBER if risk_pct < 5 else _RED)
        st.metric(
            label="Total Risk",
            value=f"{risk_pct:.2f}%",
            help="Sum of all stop-loss risks as % of portfolio. This is how much you'd "
                 "lose if every open trade hits its stop simultaneously.",
        )
        st.metric(
            label="Sizing Method",
            value=sizing_method.replace("_", " ").title(),
        )

    # Capital deployment progress bar
    st.sidebar.markdown("**Capital Deployment**")
    # Cap the bar at 100% for display
    bar_value = min(deployed_pct / 100.0, 1.0)
    st.sidebar.progress(bar_value, text=f"{deployed_pct:.1f}% deployed")

    # Risk bar — separate from deployment
    st.sidebar.markdown("**Risk Exposure**")
    risk_bar = min(risk_pct / 10.0, 1.0)  # scale: 10% risk = full bar (extreme scenario)
    st.sidebar.progress(risk_bar, text=f"{risk_pct:.2f}% at risk (stop-loss basis)")


def _render_position_slots(snapshot: PortfolioSnapshot) -> None:
    """Open positions counter with visual slot indicators."""
    st.sidebar.markdown("### 📂 Position Slots")

    open_n = snapshot.open_positions
    max_n = 5  # MAX_OPEN_POSITIONS

    # Build slot indicators: filled=🟢, empty=⬜
    slots = "🟢 " * open_n + "⬜ " * (max_n - open_n)
    remaining = snapshot.positions_remaining

    st.sidebar.markdown(f"{slots.strip()}")
    st.sidebar.caption(
        f"{open_n}/{max_n} positions open · {remaining} slot{'s' if remaining != 1 else ''} remaining"
    )

    if remaining == 0:
        st.sidebar.warning("⚠️ All position slots are filled.")
    elif remaining == 1:
        st.sidebar.info("ℹ️ One slot remaining — choose the next trade carefully.")


def _render_sector_exposure(snapshot: PortfolioSnapshot) -> None:
    """Sector exposure breakdown with limit indicators."""
    if not snapshot.sector_exposures:
        st.sidebar.markdown("### 🏭 Sector Exposure")
        st.sidebar.caption("No open positions.")
        return

    st.sidebar.markdown("### 🏭 Sector Exposure")
    portfolio_value = snapshot.portfolio_value
    sector_limit_pct = 30.0  # MAX_SECTOR_EXPOSURE_PCT × 100

    for sector, exposure in sorted(
        snapshot.sector_exposures.items(), key=lambda x: x[1], reverse=True
    ):
        pct = (exposure / portfolio_value * 100) if portfolio_value > 0 else 0
        bar_fill = min(pct / sector_limit_pct, 1.0)  # full bar = at limit

        # Colour the label based on how close to limit
        if pct >= sector_limit_pct:
            indicator = "🔴"
        elif pct >= sector_limit_pct * 0.80:
            indicator = "🟡"
        else:
            indicator = "🟢"

        st.sidebar.markdown(f"{indicator} **{sector}** — {pct:.1f}%")
        st.sidebar.progress(bar_fill, text=f"{exposure:,.0f} (limit: {sector_limit_pct:.0f}%)")


def _render_proposed_trade(symbol: str, proposed_size: dict) -> None:
    """
    Show the risk verdict for a stock the user has selected or hovered over.

    proposed_size keys:
        shares (int), position_value (float), capital_at_risk (float),
        fraction_used (float), reasoning (str), approved (bool),
        rejection_reasons (list[str])
    """
    st.sidebar.markdown(f"### 🎯 Proposed Trade: {symbol}")

    approved = proposed_size.get("approved", False)
    shares = proposed_size.get("shares", 0)
    position_value = proposed_size.get("position_value", 0.0)
    capital_at_risk = proposed_size.get("capital_at_risk", 0.0)
    fraction_used = proposed_size.get("fraction_used", 0.0)
    rejection_reasons = proposed_size.get("rejection_reasons", [])

    if approved:
        st.sidebar.success("✅ **Risk Check: APPROVED**")
    else:
        st.sidebar.error("❌ **Risk Check: REJECTED**")
        for reason in rejection_reasons:
            st.sidebar.caption(f"• {reason}")

    if shares > 0:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Shares", shares)
            st.metric("Position Size", f"{position_value:,.0f}")
        with col2:
            st.metric("Capital at Risk", f"{capital_at_risk:,.0f}")
            st.metric("Portfolio %", f"{fraction_used * 100:.1f}%")

    with st.sidebar.expander("Sizing Reasoning", expanded=False):
        st.caption(proposed_size.get("reasoning", "No reasoning available."))
