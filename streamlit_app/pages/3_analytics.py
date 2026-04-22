"""
Analytics Dashboard — Sprint 5.

Displays performance metrics, equity curve, and trade history
for all paper trades logged by the Sprint 4 execution engine.

Architecture:
  SQLiteRepository reads data/trades.db (Sprint 4 paper trades).
  PerformanceEngine computes all metrics from ClosedTrade read models.
  This page only imports from src.journal — zero coupling to src.execution.
"""

import os
import sys

import streamlit as st

# ---------------------------------------------------------------------------
# Guard: journal module availability
# ---------------------------------------------------------------------------

try:
    from src.journal.repository import SQLiteRepository
    from src.journal.analytics import PerformanceEngine

    JOURNAL_AVAILABLE = True
except ImportError as e:
    JOURNAL_AVAILABLE = False
    _IMPORT_ERROR = str(e)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Trade Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Trade Analytics")
st.caption("Performance metrics derived from Sprint 4 paper trades.")

if not JOURNAL_AVAILABLE:
    st.error(f"Journal module not available: {_IMPORT_ERROR}")
    st.info("Ensure `src/journal/` is on your PYTHONPATH and all Sprint 5 files are present.")
    st.stop()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

DB_PATH = os.environ.get("TRADES_DB_PATH", "data/trades.db")

if not os.path.exists(DB_PATH):
    st.warning(
        f"No trade database found at `{DB_PATH}`. "
        "Execute some paper trades in the main scanner page first."
    )
    st.info(
        "Paper trades are logged automatically when you click **Execute Paper Trade** "
        "on a BUY signal in the scanner."
    )
    st.stop()

repo = SQLiteRepository(db_path=DB_PATH)
trades = repo.get_closed_trades()
equity_data = repo.get_equity_curve()

if not trades:
    st.info(
        "No closed trades yet. Paper trades appear here once they are closed "
        "(stopped out or manually closed via the execution engine)."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Top-level metrics
# ---------------------------------------------------------------------------

metrics = PerformanceEngine.summary(trades)

st.subheader("Performance Summary")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_pnl = metrics["total_pnl"]
    st.metric(
        "Total P&L",
        f"₹{total_pnl:,.0f}",
        delta=f"{'▲' if total_pnl >= 0 else '▼'} {abs(total_pnl):,.0f}",
        delta_color="normal",
    )
    st.metric("Total Trades", metrics["total_trades"])

with col2:
    wr = metrics["win_rate"]
    st.metric("Win Rate", f"{wr:.1%}" if wr is not None else "—")
    st.metric("Winners / Losers", f"{metrics['winners']} / {metrics['losers']}")

with col3:
    pf = metrics["profit_factor"]
    st.metric("Profit Factor", f"{pf:.2f}" if pf is not None else "—")
    exp = metrics["expectancy"]
    st.metric("Expectancy", f"₹{exp:,.0f}" if exp is not None else "—")

with col4:
    sharpe = metrics["sharpe_ratio"]
    st.metric("Sharpe Ratio", f"{sharpe:.2f}" if sharpe is not None else "—")
    sortino = metrics["sortino_ratio"]
    st.metric("Sortino Ratio", f"{sortino:.2f}" if sortino is not None else "—")

with col5:
    dd = metrics["max_drawdown"]
    st.metric("Max Drawdown", f"₹{dd:,.0f}" if dd is not None else "—")
    avg_r = metrics["avg_r_multiple"]
    st.metric("Avg R-Multiple", f"{avg_r:.2f}R" if avg_r is not None else "—")

st.divider()

# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

st.subheader("Equity Curve")

if equity_data:
    try:
        import pandas as pd

        eq_df = pd.DataFrame(equity_data)
        eq_df["date"] = pd.to_datetime(eq_df["date"])
        eq_df = eq_df.sort_values("date")

        st.line_chart(
            eq_df.set_index("date")["cumulative_pnl"],
            width="stretch",
        )
        st.caption("Cumulative realised P&L across all closed paper trades.")
    except ImportError:
        # Fallback: display as table if pandas unavailable
        st.dataframe(equity_data, width="stretch")
else:
    st.info("No equity curve data available yet.")

st.divider()

# ---------------------------------------------------------------------------
# Per-strategy breakdown
# ---------------------------------------------------------------------------

st.subheader("Performance by Strategy")

breakdown = PerformanceEngine.by_strategy(trades)
if breakdown:
    try:
        import pandas as pd

        rows = []
        for strategy, m in sorted(breakdown.items()):
            rows.append({
                "Strategy": strategy,
                "Trades": m["trades"],
                "Win Rate": f"{m['win_rate']:.1%}" if m["win_rate"] is not None else "—",
                "Profit Factor": f"{m['profit_factor']:.2f}" if m["profit_factor"] is not None else "—",
                "Avg R": f"{m['avg_r_multiple']:.2f}" if m["avg_r_multiple"] is not None else "—",
                "Total P&L": f"₹{m['total_pnl']:,.0f}",
                "Expectancy": f"₹{m['expectancy']:,.0f}" if m["expectancy"] is not None else "—",
            })

        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    except ImportError:
        st.json(breakdown)

st.divider()

# ---------------------------------------------------------------------------
# Trade history table
# ---------------------------------------------------------------------------

st.subheader("Trade History")

_status_emoji = {"CLOSED": "✅", "STOPPED_OUT": "🛑"}

try:
    import pandas as pd

    rows = []
    for t in sorted(trades, key=lambda x: x.exit_timestamp, reverse=True):
        rows.append({
            "Symbol": t.symbol,
            "Strategy": t.strategy,
            "Status": f"{_status_emoji.get(t.status, '')} {t.status}",
            "Entry": f"₹{t.entry_price:,.2f}",
            "Exit": f"₹{t.exit_price:,.2f}",
            "Shares": t.shares,
            "P&L": f"₹{t.pnl:+,.0f}",
            "P&L %": f"{t.pnl_pct:+.2f}%",
            "R-Multiple": f"{t.r_multiple:+.2f}R",
            "Hold Days": t.hold_days,
            "Exit Date": t.exit_timestamp.strftime("%Y-%m-%d"),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)

except ImportError:
    for t in sorted(trades, key=lambda x: x.exit_timestamp, reverse=True):
        status_icon = _status_emoji.get(t.status, "")
        st.write(
            f"{status_icon} **{t.symbol}** | {t.strategy} | "
            f"P&L: ₹{t.pnl:+,.0f} ({t.pnl_pct:+.2f}%) | "
            f"{t.exit_timestamp.strftime('%Y-%m-%d')}"
        )

# ---------------------------------------------------------------------------
# Metric explanations expander
# ---------------------------------------------------------------------------

with st.expander("📖 What do these metrics mean?"):
    st.markdown("""
**Win Rate** — % of trades that closed profitably. Easy to game (lots of small wins + one
large loss). Always read alongside Profit Factor.

**Profit Factor** — Gross profit ÷ gross loss. Above 1.5 is decent; above 2.0 is good.
Below 1.0 means the strategy loses money overall.

**Expectancy** — Expected rupee P&L per trade. Positive = the strategy has mathematical edge.
This is the most honest single-number measure of strategy health.

**R-Multiple** — Each trade's P&L expressed in units of initial risk (1R = entry − stop loss).
A strategy with 40% win rate can still be profitable if average R > 1.5. More meaningful
than raw P&L because it normalises for position size.

**Sharpe Ratio** — Annualised (mean daily return − risk-free rate) ÷ std deviation × √252.
Above 1.0 is acceptable; above 2.0 is excellent. Penalises both upside and downside volatility.

**Sortino Ratio** — Like Sharpe but only penalises *downside* volatility. Better for
strategies with asymmetric return profiles. Higher than Sharpe = most volatility is upside.

**Max Drawdown** — Largest peak-to-trough decline in cumulative P&L. This is what kills
accounts psychologically. Calculated on the equity curve, not individual trades.
    """)
