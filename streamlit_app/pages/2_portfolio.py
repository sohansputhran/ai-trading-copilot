"""
Portfolio Page - View all open positions and portfolio metrics

Enhanced version with:
- Sidebar with quick actions and insights
- Red/Green color coding for profit/loss
- Better visual hierarchy
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from src.execution.order_manager import OrderManager
from src.execution.paper_broker import PaperBroker
from src.risk_management.portfolio import PortfolioRisk
from src.journal.repository import SQLiteRepository

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

# Helper function to display metrics with color coding
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
    """
    
    if delta:
        # Determine delta color
        if "↑" in delta or delta.startswith("+"):
            delta_color = "#00C853"  # Green
        elif "↓" in delta or delta.startswith("-"):
            delta_color = "#FF1744"  # Red
        else:
            delta_color = "#888"
        
        # html += f"""
        # <p style="color: {delta_color}; font-size: 14px; margin: 5px 0 0 0;">
        #     {delta}
        # </p>
        # """
    
    html += "</div>"
    
    st.markdown(html, unsafe_allow_html=True)

# Get data
order_manager = st.session_state.order_manager
portfolio = st.session_state.portfolio_risk
open_positions = order_manager.get_open_positions()
snapshot = portfolio.snapshot()

# Calculate total P&L from all positions
total_current_pnl = 0.0
for order in open_positions:
    symbol = order.symbol
    entry_price = order.fill_price or order.requested_price or 0.0
    quantity = order.shares
    
    current_price = get_current_price(symbol)
    if current_price == 0.0:
        current_price = entry_price
    
    pnl = (current_price - entry_price) * quantity
    total_current_pnl += pnl

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

# SIDEBAR

st.sidebar.title("💼 Portfolio")
st.sidebar.markdown("---")

# Quick stats in sidebar
st.sidebar.markdown("### 📊 Quick Stats")

total_portfolio_value = snapshot.portfolio_value + total_current_pnl
st.sidebar.metric(
    "Portfolio Value",
    f"₹{total_portfolio_value:,.0f}",
    delta=f"₹{total_current_pnl:+,.0f}" if total_current_pnl != 0 else None
)

st.sidebar.metric(
    "Available Capital",
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

# Calculate total P&L from all positions
total_current_pnl = 0.0
for order in open_positions:
    symbol = order.symbol
    entry_price = order.fill_price or order.requested_price or 0.0
    quantity = order.shares
    
    current_price = get_current_price(symbol)
    if current_price == 0.0:
        current_price = entry_price
    
    pnl = (current_price - entry_price) * quantity
    total_current_pnl += pnl

# Portfolio metrics row
st.markdown("## 📊 Portfolio Metrics")

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    total_portfolio_value = snapshot.portfolio_value + total_current_pnl
    st.metric(
        "Portfolio Value",
        f"₹{total_portfolio_value:,.0f}",
        delta=f"₹{total_current_pnl:+,.0f}" if total_current_pnl != 0 else None,
        help="Total portfolio value (starting capital + unrealized P&L)"
    )

with metric_col2:
    st.metric(
        "Available Capital",
        f"₹{snapshot.available_capital:,.0f}",
        help="Capital available for new positions"
    )

with metric_col3:
    st.metric(
        "Total Risk",
        f"₹{snapshot.total_capital_at_risk:,.0f}",
        help="Total risk across all open positions"
    )

st.divider()

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    # Portfolio allocation pie chart
    if open_positions:
        # Calculate current values for each position
        allocation_data = []
        for order in open_positions:
            symbol = order.symbol
            entry_price = order.fill_price or order.requested_price or 0.0
            quantity = order.shares
            
            current_price = get_current_price(symbol)
            if current_price == 0.0:
                current_price = entry_price
            
            current_value = quantity * current_price
            allocation_data.append({
                "Stock": symbol,
                "Value": current_value
            })
        
        # Add cash position
        allocation_data.append({
            "Stock": "Cash",
            "Value": snapshot.available_capital
        })
        
        allocation_df = pd.DataFrame(allocation_data)
        
        fig_allocation = px.pie(
            allocation_df,
            values="Value",
            names="Stock",
            title="Portfolio Allocation",
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
            height=400
        )
        st.plotly_chart(fig_allocation, width='stretch')
    else:
        # Show 100% cash if no positions
        fig_allocation = px.pie(
            values=[snapshot.portfolio_value],
            names=["Cash"],
            title="Portfolio Allocation",
            hole=0.4,
            color_discrete_sequence=['#90CAF9']
        )
        fig_allocation.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>'
        )
        fig_allocation.update_layout(height=400)
        st.plotly_chart(fig_allocation, width='stretch')

with viz_col2:
    # Portfolio value over time line chart
    # Build comprehensive daily portfolio value including both closed and open positions
    
    def build_daily_portfolio_timeline():
        """
        Build a daily timeline of portfolio value including:
        - Closed trades (realized P&L)
        - Open positions (unrealized P&L calculated daily)
        """
        try:
            repo = SQLiteRepository(db_path="data/trades.db")
            closed_trades = repo.get_all_trades()
            
            # Get all trades (both open and closed)
            all_entry_dates = []
            
            # Add closed trade entry dates
            for trade in closed_trades:
                all_entry_dates.append(trade.entry_timestamp)
            
            # Add open position entry dates
            for order in open_positions:
                # Use fill_timestamp (when position was opened) or created_at as fallback
                timestamp = order.fill_timestamp or order.created_at
                if timestamp:
                    all_entry_dates.append(timestamp)
            
            if not all_entry_dates:
                # No trades at all
                return None
            
            # Find earliest and latest dates
            start_date = min(all_entry_dates).date()
            end_date = datetime.now().date()
            
            # Build daily timeline - initialize all days with starting value
            daily_values = {}
            current_date = start_date
            
            while current_date <= end_date:
                daily_values[current_date] = PORTFOLIO_VALUE
                current_date += timedelta(days=1)
            
            # Process closed trades - add realized P&L on exit date
            for trade in closed_trades:
                if trade.pnl is not None and trade.exit_timestamp:
                    exit_date = trade.exit_timestamp.date()
                    # Add realized P&L from this exit date onwards
                    for date in daily_values.keys():
                        if date >= exit_date:
                            daily_values[date] += trade.pnl
            
            # Process open positions - calculate daily unrealized P&L
            position_histories = {}  # Cache historical data
            
            for order in open_positions:
                symbol = order.symbol
                entry_price = order.fill_price or order.requested_price or 0.0
                quantity = order.shares
                
                # Use fill_timestamp (when position was opened) or created_at as fallback
                entry_timestamp = order.fill_timestamp or order.created_at
                if not entry_timestamp:
                    continue
                    
                entry_date = entry_timestamp.date()
                
                # Fetch historical prices for this symbol
                if symbol not in position_histories:
                    try:
                        ticker_symbol = symbol if symbol.endswith('.NS') else f"{symbol}.NS"
                        ticker = yf.Ticker(ticker_symbol)
                        
                        # Get historical data from entry date to today
                        hist_data = ticker.history(start=entry_date, end=end_date + timedelta(days=1))
                        
                        if not hist_data.empty:
                            position_histories[symbol] = hist_data
                        else:
                            position_histories[symbol] = None
                    except:
                        position_histories[symbol] = None
                
                hist_data = position_histories[symbol]
                
                if hist_data is not None and not hist_data.empty:
                    # For each day this position was open, calculate unrealized P&L
                    for date in daily_values.keys():
                        if date >= entry_date:
                            # Find the price for this date or the most recent previous date
                            try:
                                # Convert hist_data index to dates for comparison
                                hist_dates = pd.to_datetime(hist_data.index).date
                                
                                if date in hist_dates:
                                    # Exact match
                                    price = float(hist_data.loc[hist_data.index.date == date, 'Close'].iloc[0])
                                else:
                                    # Use the most recent price before this date
                                    past_prices = hist_data[pd.to_datetime(hist_data.index).date <= date]
                                    if not past_prices.empty:
                                        price = float(past_prices['Close'].iloc[-1])
                                    else:
                                        # No historical data yet, use entry price
                                        price = entry_price
                                
                                unrealized_pnl = (price - entry_price) * quantity
                                daily_values[date] += unrealized_pnl
                            except Exception as e:
                                # If we can't get price for this date, skip
                                pass
                else:
                    # No historical data available, use current price for all dates
                    current_price = get_current_price(symbol)
                    if current_price > 0:
                        for date in daily_values.keys():
                            if date >= entry_date:
                                unrealized_pnl = (current_price - entry_price) * quantity
                                daily_values[date] += unrealized_pnl
            
            # Convert to DataFrame
            timeline_data = [
                {"Date": date, "Portfolio Value": value}
                for date, value in sorted(daily_values.items())
            ]
            
            return pd.DataFrame(timeline_data)
        
        except Exception as e:
            return None
    
    # Build the timeline with a spinner
    with st.spinner("Loading portfolio history..."):
        timeline_df = build_daily_portfolio_timeline()
    
    if timeline_df is not None and len(timeline_df) > 0:
        # Create the line chart
        fig_timeline = px.line(
            timeline_df,
            x="Date",
            y="Portfolio Value",
            title="Portfolio Value Over Time (Daily)",
            markers=True
        )
        fig_timeline.update_traces(
            line=dict(color='#2196F3', width=3),
            marker=dict(size=6),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>₹%{y:,.0f}<extra></extra>'
        )
        fig_timeline.update_layout(
            yaxis_tickprefix='₹',
            yaxis_tickformat=',.0f',
            hovermode='x unified',
            height=400,
            xaxis=dict(
                type='date',
                tickformat='%Y-%m-%d',
            )
        )
        fig_timeline.update_xaxes(title="Date")
        fig_timeline.update_yaxes(title="Portfolio Value (₹)")
        
        # Add a horizontal line at starting value for reference
        fig_timeline.add_hline(
            y=PORTFOLIO_VALUE,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            annotation_text=f"Start: ₹{PORTFOLIO_VALUE:,.0f}",
            annotation_position="right"
        )
        
        # Calculate and show total change
        start_value = timeline_df['Portfolio Value'].iloc[0]
        end_value = timeline_df['Portfolio Value'].iloc[-1]
        total_change = end_value - start_value
        total_change_pct = (total_change / start_value) * 100 if start_value > 0 else 0
        
        st.plotly_chart(fig_timeline, width='stretch')
        
        # Show summary stats below the chart
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Change", f"₹{total_change:+,.0f}", f"{total_change_pct:+.2f}%")
        with col2:
            st.metric("Days Tracked", len(timeline_df))
        with col3:
            max_value = timeline_df['Portfolio Value'].max()
            st.metric("Peak Value", f"₹{max_value:,.0f}")
        
    else:
        # Fallback - simple current value chart
        fig_timeline = go.Figure()
        current_total = snapshot.portfolio_value + total_current_pnl
        
        if open_positions:
            # Get earliest entry date from open positions
            entry_dates = []
            for order in open_positions:
                timestamp = order.fill_timestamp or order.created_at
                if timestamp:
                    entry_dates.append(timestamp)
            
            if entry_dates:
                earliest_date = min(entry_dates)
            else:
                earliest_date = datetime.now()
        else:
            earliest_date = datetime.now()
        
        dates = [earliest_date.date(), datetime.now().date()]
        values = [PORTFOLIO_VALUE, current_total]
        
        fig_timeline.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines+markers',
            line=dict(color='#2196F3', width=3),
            marker=dict(size=10),
            hovertemplate='<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>'
        ))
        
        fig_timeline.update_layout(
            title="Portfolio Value Over Time",
            yaxis_tickprefix='₹',
            yaxis_tickformat=',.0f',
            xaxis_title="Date",
            yaxis_title="Portfolio Value (₹)",
            height=400,
            showlegend=False,
            xaxis=dict(type='date')
        )
        st.plotly_chart(fig_timeline, width='stretch')

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
        colored_metric(
            label="Total P&L",
            value=total_pnl,
            prefix="₹",
            delta=f"↑ {total_pnl_pct}%",
            help_text="Total profit/loss across all closed positions"
        )
    
    with pnl_col2:
        winning = sum(1 for p in position_data if p["pnl"] >= 0)
        losing = len(position_data) - winning
        win_rate = (winning / len(position_data) * 100) if position_data else 0
        
        colored_metric(
            label="Win Rate",
            value=win_rate,
            delta=f"{winning}W / {losing}L",
            prefix= " ",
            help_text="Percentage of winning trades",
            threshold=50.0  # Green if >= 50%, Red if < 50%
        )
    
    with pnl_col3:
        avg_pnl = total_pnl / len(position_data) if position_data else 0
        colored_metric(
            label="Avg P&L per Position",
            value=avg_pnl,
            prefix="₹",
            help_text="Average profit/loss per closed position"
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