"""
Portfolio Page - View all open positions and portfolio metrics

Enhanced version with:
- Sidebar with quick actions and insights
- Red/Green color coding for profit/loss
- Better visual hierarchy
- Dummy data support for Streamlit Cloud
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# DEPLOYMENT MODE DETECTION
# ============================================================================

IS_CLOUD_DEPLOYMENT = os.environ.get("STREAMLIT_CLOUD", "false").lower() == "true"

# Try to import real modules
REAL_DATA_AVAILABLE = False
if not IS_CLOUD_DEPLOYMENT:
    try:
        import yfinance as yf
        from src.execution.order_manager import OrderManager
        from src.execution.paper_broker import PaperBroker
        from src.risk_management.portfolio import PortfolioRisk, Position
        from src.journal.repository import SQLiteRepository
        from src.journal.analytics import PerformanceEngine
        REAL_DATA_AVAILABLE = True
    except ImportError:
        IS_CLOUD_DEPLOYMENT = True
        REAL_DATA_AVAILABLE = False

# ============================================================================
# DUMMY DATA GENERATION
# ============================================================================

def generate_dummy_data():
    """Generate dummy portfolio data for Streamlit Cloud demo"""
    from dataclasses import dataclass
    import uuid
    
    @dataclass
    class DummyOrder:
        order_id: str
        symbol: str
        shares: int
        fill_price: float
        requested_price: float
        stop_loss: float
        take_profit: float
        strategy: str
        confidence: float
        capital_at_risk: float
        agent_reasoning: str
        timestamp: str
    
    dummy_orders = [
        DummyOrder(
            order_id=str(uuid.uuid4()),
            symbol="RELIANCE",
            shares=15,
            fill_price=2450.50,
            requested_price=2450.00,
            stop_loss=2400.00,
            take_profit=2550.00,
            strategy="momentum_breakout",
            confidence=0.78,
            capital_at_risk=757.50,  # (2450.50 - 2400.00) * 15 = 757.50
            agent_reasoning="Strong momentum breakout above 2440 resistance. Volume spike 2.3x average. RSI at 62 (not overbought). MACD bullish crossover. Target: 2550 (R:R = 1.98x)",
            timestamp=(datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S")
        ),
        DummyOrder(
            order_id=str(uuid.uuid4()),
            symbol="TCS",
            shares=25,
            fill_price=3680.00,
            requested_price=3680.00,
            stop_loss=3620.00,
            take_profit=3800.00,
            strategy="technical_analysis",
            confidence=0.72,
            capital_at_risk=1500.00,  # (3680.00 - 3620.00) * 25 = 1500.00
            agent_reasoning="Bouncing off 50-day EMA support at 3670. Bullish engulfing pattern on daily. Volume confirmation present. Stochastic oversold reversal. Conservative R:R = 2.0x",
            timestamp=(datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")
        ),
        DummyOrder(
            order_id=str(uuid.uuid4()),
            symbol="HDFCBANK",
            shares=12,
            fill_price=1625.00,
            requested_price=1625.00,
            stop_loss=1590.00,
            take_profit=1710.00,
            strategy="breakout_strategy",
            confidence=0.85,
            capital_at_risk=420.00,  # (1625.00 - 1590.00) * 12 = 420.00
            agent_reasoning="Breaking above multi-week consolidation range. ADX strengthening (28). Volume 1.8x average. Price action clean. Measured move target: 1710. Strong institutional buying.",
            timestamp=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        ),
        DummyOrder(
            order_id=str(uuid.uuid4()),
            symbol="INFY",
            shares=30,
            fill_price=1455.00,
            requested_price=1455.00,
            stop_loss=1430.00,
            take_profit=1510.00,
            strategy="momentum_breakout",
            confidence=0.68,
            capital_at_risk=750.00,  # (1455.00 - 1430.00) * 30 = 750.00
            agent_reasoning="Momentum building after earnings beat. Price gapping above resistance. MACD positive. Risk: sector rotation. Stop below swing low. Target previous high.",
            timestamp=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        ),
    ]
    
    # Current prices (mix of winners/losers)
    current_prices = {
        "RELIANCE": 2480.00,  # +1.2%
        "TCS": 3650.00,       # -0.8%
        "HDFCBANK": 1665.00,  # +2.5%
        "INFY": 1448.00,      # -0.5%
    }
    
    @dataclass
    class DummyClosedTrade:
        pnl: float
    
    closed_trades = [
        DummyClosedTrade(pnl=4500.0),
        DummyClosedTrade(pnl=-1200.0),
        DummyClosedTrade(pnl=8200.0),
        DummyClosedTrade(pnl=2100.0),
        DummyClosedTrade(pnl=-800.0),
    ]
    
    @dataclass
    class DummySnapshot:
        portfolio_value: float
        available_capital: float
        deployed_capital: float
        total_capital_at_risk: float
        open_positions: int
        risk_per_trade: float
        max_position_size: float
    
    # Calculate total capital at risk from all open positions
    total_capital_at_risk = sum(order.capital_at_risk for order in dummy_orders)
    
    # Create snapshot instance with calculated values
    snapshot = DummySnapshot(
        portfolio_value=500000,
        available_capital=345000,
        deployed_capital=155000,
        total_capital_at_risk=total_capital_at_risk,
        open_positions=4,
        risk_per_trade=0.015,
        max_position_size=0.05
    )
    
    # Generate portfolio value timeline (last 30 days)
    portfolio_timeline = []
    base_value = 500000
    current_date = datetime.now()
    
    for days_ago in range(30, -1, -1):
        date = current_date - timedelta(days=days_ago)
        
        # Create realistic portfolio growth with some volatility
        # Day 0 (30 days ago): 500,000
        # Day 30 (today): 512,762 (+2.55%)
        # With ups and downs in between
        
        if days_ago >= 25:
            # First week: slight growth
            value = base_value + (days_ago - 30) * 400
        elif days_ago >= 20:
            # Week 2: small dip
            value = base_value - 2000 + (25 - days_ago) * 300
        elif days_ago >= 15:
            # Week 3: recovery and growth
            value = base_value - 500 + (20 - days_ago) * 800
        elif days_ago >= 10:
            # Week 4: strong performance
            value = base_value + 3500 + (15 - days_ago) * 600
        elif days_ago >= 5:
            # Week 5: consolidation
            value = base_value + 6500 + (10 - days_ago) * 400
        else:
            # Last 5 days: final push to current value
            value = base_value + 8500 + (5 - days_ago) * 850
        
        portfolio_timeline.append({
            "Date": date,
            "Portfolio Value (₹)": round(value, 2)
        })
    
    return dummy_orders, current_prices, closed_trades, snapshot, portfolio_timeline

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_current_price(symbol: str) -> float:
    """Fetch current market price for a symbol"""
    if IS_CLOUD_DEPLOYMENT:
        return 0.0
    
    try:
        import yfinance as yf
        
        if not symbol.endswith(".NS"):
            symbol = symbol + ".NS"
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        
        if not data.empty:
            return float(data['Close'].iloc[-1])
        
        return 0.0
    except Exception as e:
        return 0.0

def colored_metric(label: str, value: float, prefix: str = "₹", postfix: str = "%",
                   delta: str = None, help_text: str = None,
                   threshold: float = 0.0):
    """
    Display a metric with colored main value based on profit/loss.
    
    Args:
        label: Metric label (e.g., "Total P&L")
        value: Numeric value (e.g., 48610)
        prefix: Currency symbol (default: "₹")
        delta: Optional delta text (e.g., "↑ 9.7%")
        help_text: Optional help tooltip
    """
    
    # Determine color based on value
    if value > threshold:
        color = "#00C853"  # Green
    elif value < 0:
        color = "#FF1744"  # Red  
    else:
        color = "#757575"  # Gray
    
    # Format the value
    if prefix == " ":
        formatted_value = f"{abs(value):,.0f}{postfix}"
    else:
        formatted_value = f"{prefix}{abs(value):,.0f}"
    
    # Build HTML
    html = f"""
    <div style="padding: 10px 0;">
        <p style="font-size: 20px; font-weight: bold; margin: 0 0 5px 0;">
            {label}
        </p>
        <p style="color: {color}; font-size: 32px; font-weight: bold; margin: 0;">
            {formatted_value}
        </p>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Portfolio - AI Trading Scanner",
    page_icon="💼",
    layout="wide"
)

# ============================================================================
# DEMO MODE BANNER
# ============================================================================

if IS_CLOUD_DEPLOYMENT:
    st.info(
        "📊 **Demo Mode** - You're viewing sample portfolio data. "
        "In a real deployment with database support, this page shows live positions and P&L."
    )

# ============================================================================
# LOAD DATA (REAL OR DUMMY)
# ============================================================================

PORTFOLIO_VALUE = 500000

if IS_CLOUD_DEPLOYMENT:
    # === DUMMY DATA MODE ===
    open_positions, current_prices, closed_trades, snapshot, portfolio_timeline = generate_dummy_data()
    total_realized_pnl = sum(trade.pnl for trade in closed_trades)
    
    # Calculate unrealized P&L
    total_unrealized_pnl = 0.0
    for order in open_positions:
        pnl = (current_prices[order.symbol] - order.fill_price) * order.shares
        total_unrealized_pnl += pnl

else:
    # === REAL DATA MODE ===
    # Initialize session state objects if needed
    if "portfolio_risk" not in st.session_state:
        st.session_state.portfolio_risk = PortfolioRisk(portfolio_value=PORTFOLIO_VALUE)

    if "order_manager" not in st.session_state:
        _paper_broker = PaperBroker(
            portfolio=st.session_state.portfolio_risk,
            slippage_bps=5,
        )
        st.session_state.order_manager = OrderManager(
            broker=_paper_broker,
            db_path="data/trades.db",
        )
    
    # Portfolio hydration - critical for multi-page apps
    if "portfolio_hydrated" not in st.session_state:
        for _order in st.session_state.order_manager.get_open_positions():
            _entry = _order.fill_price or _order.requested_price or 0.0
            st.session_state.portfolio_risk.add_position(
                Position(
                    symbol=_order.symbol,
                    shares=_order.shares,
                    entry_price=_entry,
                    stop_loss=_order.stop_loss,
                    position_value=_order.shares * _entry,
                    capital_at_risk=_order.capital_at_risk,
                    sector=None,
                    strategy=_order.strategy,
                )
            )
        st.session_state["portfolio_hydrated"] = True

    # Get data
    order_manager = st.session_state.order_manager
    portfolio = st.session_state.portfolio_risk
    open_positions = order_manager.get_open_positions()
    snapshot = portfolio.snapshot()

    # Calculate REALIZED P&L from closed trades
    repo = SQLiteRepository(db_path="data/trades.db")
    closed_trades = repo.get_closed_trades()
    total_realized_pnl = sum(trade.pnl for trade in closed_trades)

    # Calculate UNREALIZED P&L from open positions
    total_unrealized_pnl = 0.0
    current_prices = {}
    for order in open_positions:
        symbol = order.symbol
        entry_price = order.fill_price or order.requested_price or 0.0
        quantity = order.shares
        
        current_price = get_current_price(symbol)
        if current_price == 0.0:
            current_price = entry_price
        
        current_prices[symbol] = current_price
        pnl = (current_price - entry_price) * quantity
        total_unrealized_pnl += pnl
    
    # For real data mode, create simple timeline with current value
    # TODO: Expand this with historical data from database
    portfolio_timeline = [
        {
            "Date": datetime.now(),
            "Portfolio Value (₹)": snapshot.portfolio_value + total_realized_pnl + total_unrealized_pnl
        }
    ]

# ============================================================================
# CALCULATE INSIGHTS
# ============================================================================

if open_positions:
    # Calculate insights
    total_pnl = 0.0
    best_performer = None
    worst_performer = None
    best_pnl = float('-inf')
    worst_pnl = float('inf')
    
    for order in open_positions:
        symbol = order.symbol
        entry_price = order.fill_price if hasattr(order, 'fill_price') else order.requested_price
        quantity = order.shares
        
        current_price = current_prices.get(symbol, entry_price)
        pnl = (current_price - entry_price) * quantity
        total_pnl += pnl
        
        if pnl > best_pnl:
            best_pnl = pnl
            best_performer = symbol
        
        if pnl < worst_pnl:
            worst_pnl = pnl
            worst_performer = symbol

# ============================================================================
# SIDEBAR
# ============================================================================

# Quick stats in sidebar
st.sidebar.markdown("### 📊 Quick Stats")

# Calculate combined P&L for delta
combined_pnl = total_realized_pnl + total_unrealized_pnl
combined_pnl_pct = (combined_pnl / snapshot.portfolio_value) * 100 if snapshot.portfolio_value > 0 else 0.0

total_portfolio_value = snapshot.portfolio_value + total_realized_pnl + total_unrealized_pnl

# Color-coded portfolio value metric
value_color = "#00c853" if combined_pnl >= 0 else "#ff1744"  # Green for profit, red for loss
delta_symbol = "▲" if combined_pnl >= 0 else "▼"

st.sidebar.markdown(
    f"""
    <div style="padding: 10px 0;">
        <p style="font-size: 14px; color: #888; font-weight: bold; margin: 0 0 5px 0;">Portfolio Value</p>
        <div style="display: flex; align-items: baseline; gap: 8px;">
            <p style="color: {value_color}; font-size: 28px; font-weight: bold; margin: 0;">
                ₹{total_portfolio_value:,.0f}
            </p>
            <p style="color: {value_color}; font-size: 14px; margin: 0;">
                {delta_symbol} {combined_pnl_pct:+.2f}%
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    f"""
    <div style="padding: 10px 0;">
        <p style="font-size: 14px; color: #888; font-weight: bold; margin: 0 0 5px 0;">Available Capital</p>
        <div style="display: flex; align-items: baseline; gap: 8px;">
            <p style="font-size: 28px; font-weight: bold; margin: 0;">
                ₹{snapshot.available_capital:,.0f}
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# Quick actions in sidebar
st.sidebar.markdown("### 🚀 Quick Actions")

if not IS_CLOUD_DEPLOYMENT:
    if st.sidebar.button("🔄 Refresh All Prices", width="stretch"):
        st.rerun()

if st.sidebar.button("📊 View Analytics", width="stretch"):
    st.switch_page("pages/3_analytics.py")

if st.sidebar.button("🔍 Back to Scanner", width="stretch"):
    st.switch_page("app.py")

st.sidebar.markdown("---")

# Portfolio insights in sidebar
st.sidebar.markdown("### 💡 Insights")

if open_positions:
    # Display insights
    if best_performer:
        st.sidebar.success(f"🏆 **Best**: {best_performer}")
        st.sidebar.caption(f"₹{best_pnl:+,.0f}")
    
    if worst_performer and worst_pnl < 0:
        st.sidebar.error(f"⚠️ **Worst**: {worst_performer}")
        st.sidebar.caption(f"₹{worst_pnl:+,.0f}")
    
    # Risk warning
    if hasattr(snapshot, 'deployed_capital') and snapshot.portfolio_value > 0:
        risk_utilization = snapshot.deployed_capital / snapshot.portfolio_value
        if risk_utilization > 0.4:
            st.sidebar.warning("⚠️ High capital deployment")
            st.sidebar.caption(f"{risk_utilization:.0%} of portfolio deployed")

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("💼 Portfolio Overview")

if not open_positions:
    st.info("📭 No open positions. Execute paper trades from the scanner to see them here.")
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.page_link("app.py", label="🔍 Scanner", icon="📊")
        st.caption("Find new trading opportunities")
    with col2:
        st.page_link("pages/3_analytics.py", label="📈 Analytics", icon="📉")
        st.caption("View historical performance")
    with col3:
        if not IS_CLOUD_DEPLOYMENT:
            if st.button("🔄 Refresh", width="stretch"):
                st.rerun()
            st.caption("Update all position prices")
        else:
            st.caption("Demo mode active")
    
    st.stop()

# Portfolio metrics
st.markdown("## 📊 Portfolio Metrics")

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric(
        "Portfolio Value",
        f"₹{total_portfolio_value:,.0f}",
        help="Total portfolio value including open positions"
    )    

with metric_col2:
    st.metric(
        "Available Capital",
        f"₹{snapshot.available_capital:,.0f}",
        help="Capital available for new positions"
    )

with metric_col3:
    total_risk = snapshot.total_capital_at_risk if hasattr(snapshot, 'total_capital_at_risk') else 0.0
    st.metric(
        "Total Risk",
        f"₹{total_risk:,.0f}",
        help="Total capital at risk across all positions"
    )

st.divider()

# Build position data
position_data = []
for order in open_positions:
    symbol = order.symbol
    entry_price = order.fill_price if hasattr(order, 'fill_price') else order.requested_price
    quantity = order.shares
    current_price = current_prices.get(symbol, entry_price)
    
    pnl = (current_price - entry_price) * quantity
    pnl_pct = ((current_price - entry_price) / entry_price) * 100
    position_value = entry_price * quantity
    current_value = current_price * quantity
    
    position_data.append({
        "order": order,
        "symbol": symbol,
        "entry_price": entry_price,
        "quantity": quantity,
        "current_price": current_price,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "position_value": position_value,
        "current_value": current_value,
    })

# Overall P&L
total_deployed = sum(p["position_value"] for p in position_data)
total_pnl_pct = (total_pnl / total_deployed * 100) if total_deployed > 0 else 0.0

# Visualizations
viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    st.markdown("### Portfolio Allocation")
    
    # Add cash to allocation
    allocation_data = [{"Category": p["symbol"], "Value": p["current_value"]} for p in position_data]
    allocation_data.append({"Category": "Cash", "Value": snapshot.available_capital})
    
    allocation_df = pd.DataFrame(allocation_data)
    
    fig_allocation = px.pie(
        allocation_df,
        values="Value",
        names="Category",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_allocation.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>'
    )
    fig_allocation.update_layout(
        showlegend=True,
        height=400,
        margin=dict(t=0, b=0, l=0, r=0)
    )
    st.plotly_chart(fig_allocation, width='stretch')

with viz_col2:
    st.markdown("### Portfolio Value Over Time")
    
    # Use the portfolio_timeline data (from dummy data or real data)
    timeline_data = pd.DataFrame(portfolio_timeline)
    
    fig_timeline = px.line(
        timeline_data,
        x="Date",
        y="Portfolio Value (₹)",
        markers=True
    )
    fig_timeline.update_traces(
        line_color='#1f77b4',
        marker=dict(size=8)
    )
    fig_timeline.update_layout(
        height=400,
        margin=dict(t=0, b=0, l=20, r=20),
        hovermode='x unified',
        xaxis=dict(
            title="Date",
            tickformat="%b %d" if len(timeline_data) > 1 else "%b %d, %Y"
        ),
        yaxis=dict(
            title="Portfolio Value (₹)",
            tickformat="₹,.0f"
        )
    )
    st.plotly_chart(fig_timeline, width='stretch')

st.divider()

# P&L Summary
st.markdown("## 💰 Performance Summary")

pnl_col1, pnl_col2, pnl_col3 = st.columns(3)

with pnl_col1:
    colored_metric(
        label="Total P&L",
        value=total_pnl,
        prefix="₹",
        delta=f"↑ {total_pnl_pct}%",
        help_text="Total profit/loss across all open positions"
    )

with pnl_col2:
    winning = sum(1 for p in position_data if p["pnl"] >= 0)
    losing = len(position_data) - winning
    win_rate = (winning / len(position_data) * 100) if position_data else 0
    
    colored_metric(
        label="Win Rate",
        value=win_rate,
        delta=f"{winning}W / {losing}L",
        prefix=" ",
        help_text="Percentage of winning positions",
        threshold=50.0  # Green if >= 50%, Red if < 50%
    )

with pnl_col3:
    avg_pnl = total_pnl / len(position_data) if position_data else 0
    colored_metric(
        label="Avg P&L per Position",
        value=avg_pnl,
        prefix="₹",
        help_text="Average profit/loss per open position"
    )

st.divider()

# Open Positions Table
st.markdown("## 📋 Open Positions")

# Sort by P&L (best performing first)
position_data.sort(key=lambda x: x["pnl"], reverse=True)

for idx, pos in enumerate(position_data, 1):
    # Color-coded P&L for expander title
    pnl_color_emoji = "🟢" if pos['pnl'] >= 0 else "🔴"
    
    with st.expander(
        f"**{idx}. {pos['symbol']}** - "
        f"{pnl_color_emoji} "
        f"₹{pos['pnl']:+,.0f} ({pos['pnl_pct']:+.2f}%)",
        expanded=(idx == 1)  # Expand first position by default
    ):
        # Position details in columns
        detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
        
        with detail_col1:
            st.markdown("**Entry Details**")
            st.metric("Entry Price", f"₹{pos['entry_price']:,.2f}")
            st.metric("Quantity", f"{pos['quantity']}")
            st.caption(f"Position Value: ₹{pos['position_value']:,.0f}")
        
        with detail_col2:
            st.markdown("**Current Status**")
            st.metric("Current Price", f"₹{pos['current_price']:,.2f}")
            price_change = pos['current_price'] - pos['entry_price']
            
            # Color-coded price change
            if price_change >= 0:
                st.metric("Price Change", f"₹{price_change:+.2f}")
            else:
                st.metric("Price Change", f"₹{price_change:+.2f}", delta_color="inverse")
            
            st.caption(f"Current Value: ₹{pos['current_value']:,.0f}")
        
        with detail_col3:
            st.markdown("**Performance**")
            
            # Color-coded P&L display
            pnl_color = "#00c853" if pos['pnl'] >= 0 else "#ff1744"
            
            st.markdown(
                f"<div style='font-size: 24px; font-weight: bold; color: {pnl_color};'>"
                f"₹{pos['pnl']:+,.0f}"
                f"</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='font-size: 18px; font-weight: bold; color: {pnl_color};'>"
                f"{pos['pnl_pct']:+.2f}%"
                f"</div>",
                unsafe_allow_html=True
            )
        
        with detail_col4:
            st.markdown("**Risk Parameters**")
            order = pos["order"]
            st.metric("Stop Loss", f"₹{order.stop_loss:,.2f}")
            st.metric("Take Profit", f"₹{order.take_profit:,.2f}")
            rr_ratio = (order.take_profit - pos['entry_price']) / (pos['entry_price'] - order.stop_loss)
            st.caption(f"R:R = {rr_ratio:.2f}x")
        
        # Order details
        st.markdown("---")
        st.markdown("**Order Details**")
        
        order_col1, order_col2, order_col3 = st.columns(3)
        
        with order_col1:
            st.caption(f"**Order ID:** {order.order_id[:12]}...")
            st.caption(f"**Strategy:** {order.strategy}")
        
        with order_col2:
            st.caption(f"**Confidence:** {order.confidence:.0%}")
            st.caption(f"**Capital at Risk:** ₹{order.capital_at_risk:,.0f}")
        
        with order_col3:
            entry_time = order.timestamp if hasattr(order, 'timestamp') else "N/A"
            st.caption(f"**Entry Time:** {entry_time}")
        
        # AI Reasoning
        if hasattr(order, 'agent_reasoning') and order.agent_reasoning:
            with st.expander("🤖 AI Reasoning"):
                st.text(order.agent_reasoning)
        
        # Close position button (only in real mode)
        if not IS_CLOUD_DEPLOYMENT:
            st.markdown("---")
            close_col1, close_col2 = st.columns([1, 4])
            
            with close_col1:
                if st.button(
                    "🚪 Close Position",
                    key=f"close_{order.order_id}",
                    type="secondary",
                    width="stretch"
                ):
                    with st.spinner(f"Closing {pos['symbol']}..."):
                        closed = order_manager.close_position(order.order_id, reason="manual")
                    
                    if closed and closed.pnl is not None:
                        pnl_color = "#00c853" if closed.pnl >= 0 else "#ff1744"
                        st.markdown(
                            f"<div style='color: {pnl_color}; font-weight: bold;'>"
                            f"✅ Position closed: ₹{closed.pnl:+,.0f} ({closed.pnl_pct:+.2f}%) | "
                            f"{closed.r_multiple:+.2f}R"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        st.rerun()
                    else:
                        st.error("❌ Failed to close position")
            
            with close_col2:
                st.caption("⚠️ Closing will execute at current market price with slippage")

# Quick actions at bottom
st.divider()
st.markdown("## 🚀 Quick Actions")

action_col1, action_col2, action_col3 = st.columns(3)

with action_col1:
    st.page_link("app.py", label="🔍 Scanner", icon="📊")
    st.caption("Find new trading opportunities")

with action_col2:
    st.page_link("pages/3_analytics.py", label="📈 Analytics", icon="📉")
    st.caption("View historical performance")

with action_col3:
    if not IS_CLOUD_DEPLOYMENT:
        if st.button("🔄 Refresh Prices", width="stretch"):
            st.rerun()
        st.caption("Update all position prices")
    else:
        st.caption("Real-time updates in local deployment")

# Footer
st.divider()
if IS_CLOUD_DEPLOYMENT:
    st.caption("💡 **Demo Mode** - This is sample data. Deploy locally with database support for real portfolio tracking.")
else:
    st.caption("💡 **Tip:** Prices automatically update when you navigate to this page. Use the sidebar Refresh button for manual updates.")