"""
Analytics Dashboard — Sprint 5

Displays performance metrics, equity curve, and trade history
for all paper trades logged by the Sprint 4 execution engine.

**DUMMY DATA MODE** for Streamlit Cloud deployment (no SQLite support)
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
# __file__ = streamlit_app/pages/3_analytics.py
# .parent              → streamlit_app/pages/
# .parent.parent       → streamlit_app/
# .parent.parent.parent → project root  ✓
project_root = Path(__file__).parent.parent.parent
# Self-healing: verify 'src' dir exists, otherwise try one level up
if not (project_root / "src").exists():
    project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

# ============================================================================
# DEPLOYMENT MODE DETECTION - FIXED LOGIC
# ============================================================================

# Check if explicitly set to cloud mode via environment variable
IS_CLOUD_DEPLOYMENT = os.environ.get("STREAMLIT_CLOUD", "false").lower() == "true"

# If not explicitly cloud mode, try importing modules to check availability
JOURNAL_AVAILABLE = False
_IMPORT_ERROR = ""
if not IS_CLOUD_DEPLOYMENT:
    try:
        # Try to import required modules - if successful, we can use real data
        from src.journal.repository import SQLiteRepository
        from src.journal.analytics import PerformanceEngine
        # Imports successful - we have real dependencies available
        JOURNAL_AVAILABLE = True
    except ImportError as e:
        # Imports failed - switch to cloud deployment mode
        IS_CLOUD_DEPLOYMENT = True
        JOURNAL_AVAILABLE = False
        _IMPORT_ERROR = str(e)
else:
    # Explicitly in cloud mode - don't try to import
    JOURNAL_AVAILABLE = False

# ============================================================================
# DUMMY DATA GENERATION
# ============================================================================

def generate_dummy_analytics_data():
    """Generate comprehensive dummy analytics data for demo"""
    from dataclasses import dataclass
    from datetime import datetime, timedelta
    import random
    
    @dataclass
    class DummyClosedTrade:
        symbol: str
        strategy: str
        status: str
        entry_price: float
        exit_price: float
        shares: int
        pnl: float
        pnl_pct: float
        r_multiple: float
        hold_days: int
        exit_timestamp: datetime
    
    # Generate 20 realistic closed trades
    strategies = ["momentum_breakout", "technical_analysis", "breakout_strategy"]
    statuses = ["CLOSED", "STOPPED_OUT"]
    
    symbols = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "HDFC", "ICICIBANK",
        "KOTAKBANK", "BHARTIARTL", "ITC", "SBIN", "BAJFINANCE",
        "ASIANPAINT", "MARUTI", "HCLTECH", "WIPRO", "ULTRACEMCO"
    ]
    
    trades = []
    base_date = datetime.now() - timedelta(days=60)
    
    # Mix of winners and losers with realistic distribution
    trade_configs = [
        # Big winners (20%)
        {"pnl_range": (5000, 12000), "r_range": (2.0, 3.5), "count": 4, "status_prob": 1.0},
        # Medium winners (30%)
        {"pnl_range": (1500, 4500), "r_range": (1.0, 2.0), "count": 6, "status_prob": 1.0},
        # Small winners (20%)
        {"pnl_range": (300, 1200), "r_range": (0.5, 1.0), "count": 4, "status_prob": 1.0},
        # Small losers (20%)
        {"pnl_range": (-1500, -300), "r_range": (-0.8, -0.3), "count": 4, "status_prob": 0.3},
        # Medium losers (10%)
        {"pnl_range": (-3000, -1500), "r_range": (-1.5, -0.8), "count": 2, "status_prob": 0.2},
    ]
    
    trade_idx = 0
    for config in trade_configs:
        for _ in range(config["count"]):
            symbol = random.choice(symbols)
            strategy = random.choice(strategies)
            
            # Determine if stopped out
            is_stopped = random.random() < (1 - config["status_prob"])
            status = "STOPPED_OUT" if is_stopped else "CLOSED"
            
            # Generate P&L
            pnl = random.uniform(*config["pnl_range"])
            r_multiple = random.uniform(*config["r_range"])
            
            # Work backwards from P&L to prices
            shares = random.randint(5, 50)
            entry_price = random.uniform(500, 3000)
            exit_price = entry_price + (pnl / shares)
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            
            # Random hold period
            hold_days = random.randint(1, 14)
            exit_timestamp = base_date + timedelta(days=trade_idx * 3)
            
            trades.append(DummyClosedTrade(
                symbol=symbol,
                strategy=strategy,
                status=status,
                entry_price=entry_price,
                exit_price=exit_price,
                shares=shares,
                pnl=pnl,
                pnl_pct=pnl_pct,
                r_multiple=r_multiple,
                hold_days=hold_days,
                exit_timestamp=exit_timestamp
            ))
            
            trade_idx += 1
    
    # Sort by exit timestamp
    trades.sort(key=lambda t: t.exit_timestamp)
    
    # Generate equity curve
    cumulative_pnl = 0
    equity_data = []
    
    for trade in trades:
        cumulative_pnl += trade.pnl
        equity_data.append({
            "date": trade.exit_timestamp.strftime("%Y-%m-%d"),
            "cumulative_pnl": cumulative_pnl
        })
    
    return trades, equity_data

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Trade Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Trade Analytics")

if IS_CLOUD_DEPLOYMENT:
    st.info(
        "📊 **Demo Mode** - You're viewing sample analytics data. "
        "In a real deployment with database support, this page shows actual trade performance."
    )
    st.caption("Performance metrics derived from dummy paper trades.")
else:
    st.caption("Performance metrics derived from Sprint 4 paper trades.")

# ============================================================================
# LOAD DATA (REAL OR DUMMY)
# ============================================================================

if IS_CLOUD_DEPLOYMENT:
    # === DUMMY DATA MODE ===
    trades, equity_data = generate_dummy_analytics_data()
    
    # Calculate metrics manually
    if trades:
        total_pnl = sum(t.pnl for t in trades)
        winners = [t for t in trades if t.pnl >= 0]
        losers = [t for t in trades if t.pnl < 0]
        
        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0
        
        metrics = {
            "total_trades": len(trades),
            "total_pnl": total_pnl,
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(trades) if trades else None,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
            "expectancy": total_pnl / len(trades) if trades else None,
            "avg_r_multiple": sum(t.r_multiple for t in trades) / len(trades) if trades else None,
            "sharpe_ratio": 1.85,  # Dummy value
            "sortino_ratio": 2.12,  # Dummy value
            "max_drawdown": -8500.0,  # Dummy value
        }
        
        # Per-strategy breakdown
        from collections import defaultdict
        strategy_metrics = defaultdict(lambda: {"trades": [], "pnl": 0})
        
        for t in trades:
            strategy_metrics[t.strategy]["trades"].append(t)
            strategy_metrics[t.strategy]["pnl"] += t.pnl
        
        breakdown = {}
        for strategy, data in strategy_metrics.items():
            strat_trades = data["trades"]
            strat_winners = [t for t in strat_trades if t.pnl >= 0]
            strat_losers = [t for t in strat_trades if t.pnl < 0]
            
            strat_gross_profit = sum(t.pnl for t in strat_winners) if strat_winners else 0
            strat_gross_loss = abs(sum(t.pnl for t in strat_losers)) if strat_losers else 0
            
            breakdown[strategy] = {
                "trades": len(strat_trades),
                "win_rate": len(strat_winners) / len(strat_trades) if strat_trades else None,
                "profit_factor": strat_gross_profit / strat_gross_loss if strat_gross_loss > 0 else None,
                "avg_r_multiple": sum(t.r_multiple for t in strat_trades) / len(strat_trades) if strat_trades else None,
                "total_pnl": data["pnl"],
                "expectancy": data["pnl"] / len(strat_trades) if strat_trades else None,
            }

else:
    # === REAL DATA MODE ===
    if not JOURNAL_AVAILABLE:
        st.error(
            f"❌ Journal module failed to import locally.\n\n"
            f"Error: `{_IMPORT_ERROR}`\n\n"
            f"project_root resolved to: `{project_root}`\n\n"
            f"Ensure `src/journal/` is on your PYTHONPATH and all Sprint 5 files are present."
        )
        st.stop()

    # Use absolute path so it works regardless of working directory
    DB_PATH = str(project_root / "data" / "trades.db")
    
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
    
    metrics = PerformanceEngine.summary(trades)
    breakdown = PerformanceEngine.by_strategy(trades)

# ============================================================================
# TOP-LEVEL METRICS
# ============================================================================

st.subheader("Performance Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_pnl = metrics["total_pnl"]
    pnl_color = "#00c853" if total_pnl >= 0 else "#ff1744"  # Green for profit, red for loss
    delta_symbol = "▲" if total_pnl >= 0 else "▼"
    
    st.markdown(
        f"""
        <div style="padding: 10px 0;">
            <p style="font-size: 14px; color: #808495; margin: 0 0 8px 0; font-weight: bold;">Total P&L</p>
            <p style="color: {pnl_color}; font-size: 32px; font-weight: bold; margin: 0; line-height: 1;">
                ₹{total_pnl:,.0f}
            </p>
            <p style="font-size: 12px; color: #808495; margin: 8px 0 0 0;">
                {metrics['total_trades']} trades
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    wr = metrics["win_rate"]
    st.markdown(
        f"""
        <div style="padding: 10px 0;">
            <p style="font-size: 14px; color: #808495; margin: 0 0 8px 0; font-weight: bold;">Win Rate</p>
            <p style="color: #262730; font-size: 32px; font-weight: bold; margin: 0; line-height: 1;">
                {f"{wr:.1%}" if wr is not None else "—"}
            </p>
            <p style="font-size: 12px; color: #808495; margin: 8px 0 0 0;">
                {metrics['winners']} wins / {metrics['losers']} losses
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    pf = metrics["profit_factor"]
    if pf is not None:
        # Color based on profit factor quality
        if pf > 3:
            pf_label = "🔥 Excellent"
            pf_color = "#00c853"  # Green - Excellent
        elif pf > 2:
            pf_label = "✓ Very Good"
            pf_color = "#4caf50"  # Light green - Very Good
        elif pf > 1.5:
            pf_label = "✓ Good"
            pf_color = "#66bb6a"  # Lighter green - Good
        elif pf > 1.0:
            pf_label = "⚠️ Needs Work"
            pf_color = "#ff9800"  # Orange - Needs work
        else:
            pf_label = "❌ Poor"
            pf_color = "#ff1744"  # Red - Poor (losing system)
        
        st.markdown(
            f"""
            <div style="padding: 10px 0;">
                <p style="font-size: 14px; color: #808495; margin: 0 0 8px 0; font-weight: bold;">Profit Factor</p>
                <p style="color: {pf_color}; font-size: 32px; font-weight: bold; margin: 0; line-height: 1;">
                    {pf:.2f}
                </p>
                <p style="font-size: 12px; color: #808495; margin: 8px 0 0 0;">
                    {pf_label}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="padding: 10px 0;">
                <p style="font-size: 14px; color: #808495; margin: 0 0 8px 0; font-weight: bold;">Profit Factor</p>
                <p style="color: #262730; font-size: 32px; font-weight: bold; margin: 0; line-height: 1;">—</p>
            </div>
            """,
            unsafe_allow_html=True
        )

with col4:
    dd = metrics["max_drawdown"]
    if dd is not None:
        # Calculate as percentage of starting capital (assumes ₹500k starting)
        # For real data, this should use actual starting capital
        dd_pct = abs((dd / 500000) * 100)
        
        st.markdown(
            f"""
            <div style="padding: 10px 0;">
                <p style="font-size: 14px; color: #808495; margin: 0 0 8px 0; font-weight: bold;">Max Drawdown</p>
                <p style="color: #262730; font-size: 32px; font-weight: bold; margin: 0; line-height: 1;">
                    ₹{dd:,.0f}
                </p>
                <p style="font-size: 12px; color: #808495; margin: 8px 0 0 0;">
                    {dd_pct:.1f}% of capital
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="padding: 10px 0;">
                <p style="font-size: 14px; color: #808495; margin: 0 0 8px 0; font-weight: bold;">Max Drawdown</p>
                <p style="color: #262730; font-size: 32px; font-weight: bold; margin: 0; line-height: 1;">—</p>
            </div>
            """,
            unsafe_allow_html=True
        )

st.divider()

# ============================================================================
# EQUITY CURVE
# ============================================================================

st.subheader("Equity Curve")

if equity_data:
    try:
        import pandas as pd
        
        eq_df = pd.DataFrame(equity_data)
        eq_df["date"] = pd.to_datetime(eq_df["date"])
        eq_df = eq_df.sort_values("date")
        
        st.line_chart(
            eq_df.set_index("date")["cumulative_pnl"],
            width='stretch',
        )
        st.caption("Cumulative realised P&L across all closed paper trades.")
    except ImportError:
        st.dataframe(equity_data, width='stretch')
else:
    st.info("No equity curve data available yet.")

st.divider()

# ============================================================================
# PER-STRATEGY BREAKDOWN
# ============================================================================

st.subheader("Performance by Strategy")

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
        
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    except ImportError:
        st.json(breakdown)

st.divider()

# ============================================================================
# TRADE HISTORY TABLE
# ============================================================================

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
    st.dataframe(df, width='stretch', hide_index=True)

except ImportError:
    for t in sorted(trades, key=lambda x: x.exit_timestamp, reverse=True):
        status_icon = _status_emoji.get(t.status, "")
        st.write(
            f"{status_icon} **{t.symbol}** | {t.strategy} | "
            f"P&L: ₹{t.pnl:+,.0f} ({t.pnl_pct:+.2f}%) | "
            f"{t.exit_timestamp.strftime('%Y-%m-%d')}"
        )

# ============================================================================
# METRIC EXPLANATIONS
# ============================================================================

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

# ============================================================================
# FOOTER
# ============================================================================

st.divider()

if IS_CLOUD_DEPLOYMENT:
    st.caption(
        "💡 **Demo Mode** - This is sample data to demonstrate analytics capabilities. "
        "Deploy locally with database support to track real trade performance."
    )
else:
    st.caption("💡 **Tip:** Trade analytics update automatically as you close positions via the Portfolio page.")
