"""
Portfolio Page - View all open positions and portfolio metrics

Enhanced version with:
- Sidebar with quick actions and insights
- Red/Green color coding for profit/loss
- Better visual hierarchy
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.execution.order_manager import OrderManager
from src.execution.paper_broker import PaperBroker
from src.risk_management.portfolio import PortfolioRisk

# Initialize session state objects if needed
PORTFOLIO_VALUE = 500000

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

# Helper function to get current prices
def get_current_price(symbol: str) -> float:
    """Fetch current market price for a symbol"""
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


# Page config
st.set_page_config(
    page_title="Portfolio - AI Trading Scanner",
    page_icon="💼",
    layout="wide"
)

# Get data
order_manager = st.session_state.order_manager
portfolio = st.session_state.portfolio_risk
open_positions = order_manager.get_open_positions()
snapshot = portfolio.snapshot()

# SIDEBAR

st.sidebar.title("💼 Portfolio")
st.sidebar.markdown("---")

# Quick stats in sidebar
st.sidebar.markdown("### 📊 Quick Stats")

num_positions = len(open_positions)
max_positions = 5
slots_remaining = max_positions - num_positions

st.sidebar.metric(
    "Position Slots",
    f"{num_positions}/{max_positions}",
    f"{slots_remaining} remaining"
)

st.sidebar.metric(
    "Deployed",
    f"₹{snapshot.total_position_value:,.0f}",
    f"{snapshot.total_deployed_pct * 100:.1f}%"
)

st.sidebar.metric(
    "Available",
    f"₹{snapshot.available_capital:,.0f}"
)

st.sidebar.markdown("---")

# Quick actions in sidebar
st.sidebar.markdown("### 🚀 Quick Actions")

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
    # Calculate insights
    total_pnl = 0.0
    best_performer = None
    worst_performer = None
    best_pnl = float('-inf')
    worst_pnl = float('inf')
    
    for order in open_positions:
        symbol = order.symbol
        entry_price = order.fill_price or order.requested_price or 0.0
        quantity = order.shares
        
        current_price = get_current_price(symbol)
        if current_price == 0.0:
            current_price = entry_price
        
        pnl = (current_price - entry_price) * quantity
        total_pnl += pnl
        
        if pnl > best_pnl:
            best_pnl = pnl
            best_performer = symbol
        
        if pnl < worst_pnl:
            worst_pnl = pnl
            worst_performer = symbol
    
    # Display insights
    if best_performer:
        st.sidebar.success(f"🏆 **Best**: {best_performer}")
        st.sidebar.caption(f"₹{best_pnl:+,.0f}")
    
    if worst_performer and worst_pnl < 0:
        st.sidebar.error(f"⚠️ **Worst**: {worst_performer}")
        st.sidebar.caption(f"₹{worst_pnl:+,.0f}")
    
    # Risk warning
    risk_pct = snapshot.total_risk_pct * 100
    if risk_pct > 3.0:
        st.sidebar.warning(f"⚠️ High risk exposure: {risk_pct:.2f}%")
    
    # Capital deployment warning
    deployed_pct = snapshot.total_deployed_pct * 100
    if deployed_pct > 80:
        st.sidebar.warning(f"⚠️ {deployed_pct:.0f}% capital deployed")
else:
    st.sidebar.info("No positions to analyze")
    st.sidebar.caption("Open positions via Scanner")

st.sidebar.markdown("---")
st.sidebar.caption("💡 Prices update automatically")

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("💼 Portfolio Dashboard")
st.markdown("*Real-time view of your paper trading positions*")

# Portfolio metrics row
st.markdown("## 📊 Portfolio Metrics")

metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

with metric_col1:
    st.metric(
        "Portfolio Value",
        f"₹{snapshot.portfolio_value:,.0f}",
        help="Total capital available"
    )

with metric_col2:
    st.metric(
    label="Capital Deployed",
    value=f"₹{snapshot.total_position_value:,}",
    delta=f"{snapshot.total_deployed_pct:+.1f}%",  # Added + sign
    delta_color="normal"  # This makes it GREEN/RED
)

with metric_col3:
    st.metric(
        "Available Capital",
        f"₹{snapshot.available_capital:,.0f}",
        help="Capital available for new positions"
    )

with metric_col4:
    st.metric(
        "Total Risk",
        f"₹{snapshot.total_capital_at_risk:,.0f}",
        f"{snapshot.total_risk_pct * 100:.2f}%",
        delta_color="off"  # Neutral color
    )

with metric_col5:
    st.metric(
        "Open Positions",
        f"{len(open_positions)}/5",
        help="Number of open positions (max 5)"
    )

st.divider()

# Check if there are open positions
if not open_positions:
    st.info("📭 **No open positions yet**")
    st.markdown("""
    You don't have any open positions right now.
    
    **To open a position:**
    1. Go to the **AI Trading Scanner** page
    2. Run a scan to find opportunities
    3. Click the **Paper Trade** button on a BUY signal
    
    Your positions will appear here once you execute a trade.
    """)
    
    # Add link to scanner
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.page_link("app.py", label="🔍 Go to Scanner", icon="🚀")
    
else:
    # Calculate total P&L across all positions
    total_pnl = 0.0
    total_pnl_pct = 0.0
    position_data = []
    
    for order in open_positions:
        symbol = order.symbol
        entry_price = order.fill_price or order.requested_price or 0.0
        quantity = order.shares
        
        # Get current price
        current_price = get_current_price(symbol)
        if current_price == 0.0:
            current_price = entry_price
        
        # Calculate P&L
        pnl = (current_price - entry_price) * quantity
        pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
        position_value = quantity * entry_price
        current_value = quantity * current_price
        
        total_pnl += pnl
        
        position_data.append({
            "order": order,
            "symbol": symbol,
            "quantity": quantity,
            "entry_price": entry_price,
            "current_price": current_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "position_value": position_value,
            "current_value": current_value,
        })
    
    # Overall P&L
    total_deployed = sum(p["position_value"] for p in position_data)
    total_pnl_pct = (total_pnl / total_deployed * 100) if total_deployed > 0 else 0.0
    
    # P&L Summary
    st.markdown("## 💰 Performance Summary")
    
    pnl_col1, pnl_col2, pnl_col3 = st.columns(3)
    
    with pnl_col1:
        # Color-coded P&L metric
        if total_pnl >= 0:
            # Profit - show as positive with normal delta color
            st.metric(
                "Total P&L",
                f"₹{total_pnl:+,.0f}",
                f"{total_pnl_pct:+.2f}%"
            )
        else:
            # Loss - show as negative with inverse delta color  
            st.metric(
                "Total P&L",
                f"₹{total_pnl:+,.0f}",
                f"{total_pnl_pct:+.2f}%",
                delta_color="inverse"
            )
    
    with pnl_col2:
        winning = sum(1 for p in position_data if p["pnl"] >= 0)
        losing = len(position_data) - winning
        win_rate = (winning / len(position_data) * 100) if position_data else 0
        st.metric(
            "Win Rate",
            f"{win_rate:.0f}%",
            f"{winning}W / {losing}L",
            delta_color="off"
        )
    
    with pnl_col3:
        avg_pnl = total_pnl / len(position_data) if position_data else 0
        if avg_pnl >= 0:
            st.metric(
                "Avg P&L per Position",
                f"₹{avg_pnl:+,.0f}"
            )
        else:
            st.metric(
                "Avg P&L per Position",
                f"₹{avg_pnl:+,.0f}",
                delta_color="inverse"
            )
    
    st.divider()
    
    # Visualizations
    st.markdown("## 📈 Portfolio Visualizations")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Position allocation pie chart
        allocation_df = pd.DataFrame([
            {"Symbol": p["symbol"], "Value": p["current_value"]} 
            for p in position_data
        ])
        
        fig_allocation = px.pie(
            allocation_df,
            values="Value",
            names="Symbol",
            title="Portfolio Allocation by Stock",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_allocation.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_allocation, width="stretch")
    
    with viz_col2:
        # P&L by position with proper color coding
        pnl_df = pd.DataFrame([
            {
                "Symbol": p["symbol"],
                "P&L": p["pnl"],
                "Color": "Profit" if p["pnl"] >= 0 else "Loss"
            }
            for p in position_data
        ])
        
        fig_pnl = px.bar(
            pnl_df,
            x="Symbol",
            y="P&L",
            color="Color",
            title="P&L by Position",
            color_discrete_map={"Profit": "#00c853", "Loss": "#ff1744"}
        )
        fig_pnl.update_layout(showlegend=False)
        st.plotly_chart(fig_pnl, width="stretch")
    
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
            
            # Close position button
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
    if st.button("🔄 Refresh Prices", width="stretch"):
        st.rerun()
    st.caption("Update all position prices")

# Footer
st.divider()
st.caption("💡 **Tip:** Prices automatically update when you navigate to this page. Use the sidebar Refresh button for manual updates.")