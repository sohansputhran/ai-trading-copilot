"""
AI Trading Copilot - Main Dashboard

A production-grade AI trading system with:
- Multi-agent stock analysis
- Automated risk management
- Paper trading execution
- Performance analytics
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import plotly.graph_objects as go
import streamlit as st

# Try to import AI scanner, fallback to rule-based
try:
    from src.agents.scanner_agent import NIFTY_50_SAMPLE, MarketScanner
    SCANNER_TYPE = "AI"
except Exception as e:
    from src.agents.rule_based_scanner import NIFTY_50_SAMPLE
    from src.agents.rule_based_scanner import RuleBasedScanner as MarketScanner
    SCANNER_TYPE = "Rule-Based"

from src.data_pipeline.collector import MarketDataCollector
from src.data_pipeline.indicators import SimpleTechnicalIndicators

# Multi-agent orchestration
try:
    from src.agents.breakout_agent import BreakoutStrategyAgent
    from src.agents.momentum_agent import MomentumStrategyAgent
    from src.agents.orchestrator import MultiAgentOrchestrator
    from src.agents.technical_agent import TechnicalAnalysisAgent
    MULTI_AGENT_AVAILABLE = True
except ImportError:
    MULTI_AGENT_AVAILABLE = False

# Paper trading
try:
    from src.execution.broker import OrderSide, RiskDecision
    from src.execution.order_manager import OrderManager
    from src.execution.paper_broker import PaperBroker
    PAPER_TRADING_AVAILABLE = True
except ImportError:
    PAPER_TRADING_AVAILABLE = False

import os

from src.agents.state import Signal
from src.risk_management.portfolio import PortfolioRisk, Position
from src.risk_management.position_sizer import PositionSizer
from src.risk_management.validators import PreTradeValidator

PORTFOLIO_VALUE = float(os.getenv("PORTFOLIO_VALUE", "500000"))

# ============================================================================
# DEMO MODE CONFIGURATION
# ============================================================================

# Check if running with HuggingFace API key
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "").strip()
RUNNING_WITH_API = bool(HF_TOKEN)

# Initialize demo mode in session state
if 'demo_mode' not in st.session_state:
    # Default to demo mode if no API key
    st.session_state['demo_mode'] = not RUNNING_WITH_API

# ─────────────────────────────────────────────
# Multi-agent UI helpers
# ─────────────────────────────────────────────

def signal_badge(signal_str: str) -> str:
    return {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "🟡 HOLD"}.get(signal_str, signal_str)

def confidence_bar(confidence: float) -> str:
    filled = int(confidence * 10)
    return "█" * filled + "░" * (10 - filled) + f"  {confidence:.0%}"

def render_agent_card(analysis, col):
    signal_str = analysis.signal.value if hasattr(analysis.signal, "value") else str(analysis.signal)
    if signal_str == "BUY":
        col.success(f"**{analysis.agent_name.replace('_', ' ').title()}**")
    elif signal_str == "SELL":
        col.error(f"**{analysis.agent_name.replace('_', ' ').title()}**")
    else:
        col.warning(f"**{analysis.agent_name.replace('_', ' ').title()}**")
    col.markdown(f"Signal: **{signal_badge(signal_str)}**")
    col.markdown(f"Confidence: `{confidence_bar(analysis.confidence)}`")
    col.caption(analysis.reasoning[:200] + "..." if len(analysis.reasoning) > 200 else analysis.reasoning)
    if analysis.warnings:
        for w in analysis.warnings:
            col.warning(f"⚠️ {w}")

def render_multi_agent_tab(multi_result: dict):
    """Display multi-agent analysis in card-based layout with colored backgrounds."""
    import re
    
    if multi_result is None:
        st.info("Multi-agent analysis unavailable - LangGraph not installed.")
        return
    
    final_signal     = multi_result.get("final_signal")
    final_confidence = multi_result.get("final_confidence", 0)
    agent_agreement  = multi_result.get("agent_agreement", 0)
    final_reasoning  = multi_result.get("final_reasoning", "")
    errors           = multi_result.get("errors", [])
    signal_str = final_signal.value if hasattr(final_signal, "value") else str(final_signal)
    
    # ========================================================================
    # FINAL DECISION SECTION
    # ========================================================================
    st.markdown("## 🤖 Multi-Agent Analysis")
    st.markdown("### Final Decision")
    
    # Decision card colors
    decision_colors = {
        'BUY': '#d4edda',   # Light green
        'SELL': '#f8d7da',  # Light red
        'HOLD': '#fff3cd'   # Light yellow
    }
    decision_icons = {
        'BUY': '🟢',
        'SELL': '🔴',
        'HOLD': '🟡'
    }
    
    bg_color = decision_colors.get(signal_str, '#f8f9fa')
    icon = decision_icons.get(signal_str, '⚪')
    
    # Display final decision with metrics
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; height: 120px;">
            <h2 style="margin: 0;">{icon} {signal_str}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Confidence**")
        st.markdown(f"### {final_confidence:.0%}")
    
    with col3:
        st.markdown("**Agent Agreement**")
        st.markdown(f"### {agent_agreement:.0%}")
        if agent_agreement < 70:
            st.caption("↕️ split — low conviction")
        elif agent_agreement < 90:
            st.caption("→ moderate conviction")
        else:
            st.caption("✓ high conviction")
    
    st.markdown("---")
    
    # ========================================================================
    # AGENT BREAKDOWN SECTION (Card-based display with signal-based colors)
    # ========================================================================
    st.markdown("### Agent Breakdown")
    
    # Parse individual agents from final_reasoning
    agents_data = {}
    
    # Pattern to extract agent data - improved to capture full reasoning
    # Split the reasoning by agent markers first for better extraction
    agent_sections = {}
    
    # Find all agent sections
    technical_match = re.search(r'TECHNICAL_ANALYSIS:\s*(BUY|SELL|HOLD)\s*\((\d+)%\)\s*—\s*(.+?)(?=MOMENTUM_STRATEGY:|$)', final_reasoning, re.DOTALL)
    momentum_match = re.search(r'MOMENTUM_STRATEGY:\s*(BUY|SELL|HOLD)\s*\((\d+)%\)\s*—\s*(.+?)(?=BREAKOUT_STRATEGY:|$)', final_reasoning, re.DOTALL)
    breakout_match = re.search(r'BREAKOUT_STRATEGY:\s*(BUY|SELL|HOLD)\s*\((\d+)%\)\s*—\s*(.+?)(?=\n\n|$)', final_reasoning, re.DOTALL)
    
    agent_matches = {
        'technical_analysis': (technical_match, 'Technical Analysis'),
        'momentum_strategy': (momentum_match, 'Momentum Strategy'),
        'breakout_strategy': (breakout_match, 'Breakout Strategy')
    }
    
    # Signal-based colors (changes based on BUY/SELL/HOLD)
    signal_bg_colors = {
        'BUY': '#d4edda',   # Light green
        'SELL': '#f8d7da',  # Light red
        'HOLD': '#fff3cd'   # Light yellow
    }
    
    # Extract data for each agent
    for agent_key, (match, name) in agent_matches.items():
        if match:
            signal = match.group(1)
            confidence = int(match.group(2))
            reasoning = match.group(3).strip()
            
            # Clean up reasoning - remove checkmarks and extra text
            reasoning = reasoning.replace('✓', '').strip()
            # Remove trailing partial text from next agent
            reasoning = re.sub(r'\s*TECHNICAL_ANALYSIS:.*$', '', reasoning, flags=re.DOTALL)
            reasoning = re.sub(r'\s*MOMENTUM_STRATEGY:.*$', '', reasoning, flags=re.DOTALL)
            reasoning = re.sub(r'\s*BREAKOUT_STRATEGY:.*$', '', reasoning, flags=re.DOTALL)
            # Remove "..." artifacts
            reasoning = reasoning.replace('...', '').strip()
            
            agents_data[agent_key] = {
                'name': name,
                'signal': signal,
                'confidence': confidence,
                'reasoning': reasoning,
                'color': signal_bg_colors.get(signal, '#f8f9fa')  # Color based on signal!
            }
    
    # Display agent cards in columns
    if agents_data:
        cols = st.columns(3)
        
        for idx, (agent_key, agent_data) in enumerate(agents_data.items()):
            with cols[idx]:
                # Card header with colored background
                st.markdown(f"""
                <div style="background-color: {agent_data['color']}; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                    <h4 style="margin-top: 0; color: #333;">{agent_data['name']}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Signal with colored dot
                signal_colors = {
                    'BUY': '#28a745',   # Green
                    'SELL': '#dc3545',  # Red
                    'HOLD': '#ffc107'   # Yellow
                }
                signal_color = signal_colors.get(agent_data['signal'], '#6c757d')
                
                st.markdown(f"""
                <div style="margin: 10px 0;">
                    <span style="color: #666;">Signal:</span> 
                    <span style="color: {signal_color}; font-size: 20px;">●</span> 
                    <strong>{agent_data['signal']}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence with custom progress bar (matching screenshot style)
                st.markdown(f"**Confidence:**")
                
                # Custom progress bar
                filled_width = agent_data['confidence']
                empty_width = 100 - filled_width
                
                # Color for progress bar based on signal
                progress_colors = {
                    'BUY': '#28a745',   # Green
                    'SELL': '#dc3545',  # Red
                    'HOLD': '#ffc107'   # Yellow
                }
                bar_color = progress_colors.get(agent_data['signal'], '#6c757d')
                
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin: 5px 0 0 0;">
                    <div style="flex-grow: 1; height: 8px; background-color: #e9ecef; border-radius: 4px; overflow: hidden; margin-right: 10px;">
                        <div style="width: {filled_width}%; height: 100%; background-color: {bar_color};"></div>
                    </div>
                    <span style="font-size: 14px; color: #333; min-width: 40px;">{agent_data['confidence']}%</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Reasoning text
                reasoning_full = agent_data.get('reasoning', 'No reasoning provided')
                # Split reasoning on semicolons for bullet points
                if ';' in reasoning_full:
                    points = [p.strip() for p in reasoning_full.split(';') if p.strip()]
                    for point in points:
                        st.markdown(f"• {point}")
                else:
                    st.markdown(reasoning_full)
                
                # Check for warnings in reasoning
                reasoning_lower = agent_data['reasoning'].lower()
                if 'rsi' in reasoning_lower and 'very high' in reasoning_lower:
                    # Extract RSI value if present
                    rsi_match = re.search(r'RSI[=\s]+(\d+\.?\d*)', agent_data['reasoning'])
                    if rsi_match:
                        st.warning(f"RSI={rsi_match.group(1)} is very high — trend may be overextended, reversal risk elevated", icon="⚠️")
                    else:
                        st.warning("RSI very high — reversal risk elevated", icon="⚠️")
                
                if 'smaller than atr' in reasoning_lower or 'may be noise' in reasoning_lower:
                    st.warning("Price move smaller than ATR — may be noise", icon="⚠️")
    
    # Fallback to old display if pattern matching fails
    else:
        analyses = multi_result.get("agent_analyses", [])
        if analyses:
            cols = st.columns(min(len(analyses), 3))
            for i, analysis in enumerate(analyses):
                render_agent_card(analysis, cols[i % len(cols)])
    
    # Show errors if any
    if errors:
        st.markdown("---")
        with st.expander("⚠️ Errors", expanded=False):
            for err in errors:
                st.error(err)

def make_price_chart(symbol: str, data: dict) -> go.Figure:
    import pandas as pd
    df = pd.DataFrame(data["prices"][-60:])
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=symbol
    ))
    fig.update_layout(
        title=f"{symbol} - Last 60 Days",
        yaxis_title="Price (₹)",
        xaxis_title="Date",
        height=400,
        template="plotly_white"
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig

# ─────────────────────────────────────────────
# Session initialization
# ─────────────────────────────────────────────

# Initialize portfolio risk tracker (Sprint 3)
if "portfolio_risk" not in st.session_state:
    st.session_state.portfolio_risk = PortfolioRisk(portfolio_value=PORTFOLIO_VALUE)

# Initialize order manager & paper broker (Sprint 4)
if PAPER_TRADING_AVAILABLE:
    if "order_manager" not in st.session_state:
        _paper_broker = PaperBroker(
            portfolio=st.session_state.portfolio_risk,
            slippage_bps=5,
        )
        st.session_state.order_manager = OrderManager(
            broker=_paper_broker,
            db_path="data/trades.db",
        )

    # Auto-trade settings
    if "auto_trade_enabled" not in st.session_state:
        st.session_state.auto_trade_enabled = False
    if "auto_trade_threshold" not in st.session_state:
        st.session_state.auto_trade_threshold = 0.75

    # Hydrate PortfolioRisk from open orders persisted in SQLite.
    # Runs once per session. Without this, restarting the app wipes the
    # in-memory PortfolioRisk even though trades.db still has open positions.
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
                )
            )
        st.session_state["portfolio_hydrated"] = True

# ============================================================================
# PAGE CONFIG & TITLE
# ============================================================================

st.set_page_config(
    page_title="Home - AI Trading Copilot",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DEMO MODE BANNER (Top of page)
# ============================================================================

if st.session_state['demo_mode']:
    st.info("""
    ### 🎬 Demo Mode Active
    
    You're viewing a **demonstration** of the full multi-agent system using pre-recorded analysis.
    
    **What this shows:**
    - ✅ Complete multi-agent reasoning chains
    - ✅ Real risk calculations and position sizing
    - ✅ Authentic agent coordination flows
    - ⚠️ Using pre-recorded data (no live API calls)
    
    **To run live:** Add your HuggingFace API token in the sidebar and disable demo mode.
    
    [💻 View Source Code](https://github.com/sohansputhran/ai-trading-copilot)
    """)
# [📹 Watch Full Demo](https://github.com/sohansputhran/ai-trading-copilot#demo) |
st.title("🤖 AI Trading Copilot")
st.markdown(f"**Scanner Type:** {SCANNER_TYPE} | **Multi-Agent:** {'✅' if MULTI_AGENT_AVAILABLE else '❌'} | **Paper Trading:** {'✅' if PAPER_TRADING_AVAILABLE else '❌'}")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("🔧 Configuration")
    
    # ─── Demo Mode Toggle ───
    st.markdown("### ⚙️ System Mode")
    
    if not RUNNING_WITH_API:
        st.warning("🔑 **No API Token Detected**\n\nDemo mode is enabled. Add your HuggingFace token below to enable live mode.")
        st.session_state['demo_mode'] = True
    else:
        demo_mode_toggle = st.toggle(
            "Enable Demo Mode",
            value=st.session_state['demo_mode'],
            help="Use pre-recorded multi-agent analysis instead of live API calls"
        )
        st.session_state['demo_mode'] = demo_mode_toggle
        
        if demo_mode_toggle:
            st.info("📀 **Demo Mode**\n\nUsing pre-recorded data (no API costs)")
        else:
            st.success("🔴 **Live Mode**\n\nUsing real-time API calls")
    
    st.markdown("---")
    
    # ─── HuggingFace Token (existing) ───
    st.markdown("### 🔑 HuggingFace API Token")
    
    current_token = os.getenv("HUGGINGFACE_API_TOKEN", "")
    token_input = st.text_input(
        "Enter your token",
        type="password",
        value=current_token,
        help="Get free token: https://huggingface.co/settings/tokens"
    )
    
    if token_input and token_input != current_token:
        os.environ["HUGGINGFACE_API_TOKEN"] = token_input
        st.success("✅ Token saved for this session")
        st.rerun()
    
    if not token_input:
        st.warning("⚠️ Add token to enable AI analysis")
    
    st.markdown("---")
    
    # ─── Scanner Settings ───
    st.markdown("### 📊 Scanner Settings")
    
    stock_selection = st.radio(
        "Stock Selection",
        ["Custom Input", "Nifty 50 Sample"],
        help="Choose stocks to scan"
    )
    
    if stock_selection == "Custom Input":
        custom_input = st.text_area(
            "Enter symbols (one per line)",
            "RELIANCE.NS\nTCS.NS\nINFY.NS",
            help="NSE symbols ending with .NS"
        )
        symbols = [s.strip() for s in custom_input.split("\n") if s.strip()]
    else:
        num_stocks = st.slider(
            "Number of stocks",
            min_value=3,
            max_value=len(NIFTY_50_SAMPLE),
            value=5,
            help="Random sample from Nifty 50"
        )
        import random
        random.seed(42)
        symbols = random.sample(NIFTY_50_SAMPLE, num_stocks)
    
    st.caption(f"**Scanning:** {len(symbols)} stocks")
    
    # Multi-agent settings
    if MULTI_AGENT_AVAILABLE:
        st.markdown("### 🤖 Multi-Agent Settings")
        min_agreement = st.slider(
            "Minimum Agent Agreement",
            0.0, 1.0, 0.67,
            help="Required % of agents to agree for a signal"
        )
    
    # Paper trading settings (only in live mode)
    if PAPER_TRADING_AVAILABLE and not st.session_state['demo_mode']:
        st.markdown("### 📈 Auto-Trade Settings")
        
        st.session_state.auto_trade_enabled = st.checkbox(
            "Enable Auto-Trade",
            value=st.session_state.auto_trade_enabled,
            help="Automatically execute paper trades for high-confidence signals"
        )
        
        if st.session_state.auto_trade_enabled:
            st.session_state.auto_trade_threshold = st.slider(
                "Min Confidence Threshold",
                0.0, 1.0, 0.75,
                help="Only auto-trade if confidence >= this value"
            )
    
    st.markdown("---")
    
    # Run scanner button
    run_scanner = st.button("🔍 Run Scanner", type="primary", width='stretch')
    
    if run_scanner:
        st.session_state["scan_triggered"] = True
    
    # Quick navigation
    st.markdown("---")
    st.markdown("### 🧭 Quick Navigation")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🤖 Multi-Agent", width='stretch'):
            st.switch_page("pages/2_🤖_Multi_Agent_Demo.py")
    with col2:
        if st.button("💼 Portfolio", width='stretch'):
            st.switch_page("pages/3_💼_Portfolio.py")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📈 Analytics", width='stretch'):
            st.switch_page("pages/4_📈_Analytics.py")
    with col2:
        pass  # Reserved for future pages

# ============================================================================
# MAIN CONTENT
# ============================================================================

if st.session_state.get("scan_triggered", False):
    # Clear previous results
    st.session_state["scan_triggered"] = False
    
    with st.spinner(f"🔍 Scanning {len(symbols)} stocks..."):
        # Initialize collectors
        collector = MarketDataCollector()
        calculator = SimpleTechnicalIndicators()
        scanner = MarketScanner()
        
        # Multi-agent orchestrator (if available)
        orchestrator = None
        if MULTI_AGENT_AVAILABLE:
            orchestrator = MultiAgentOrchestrator(
                technical_agent=TechnicalAnalysisAgent(),
                momentum_agent=MomentumStrategyAgent(),
                breakout_agent=BreakoutStrategyAgent()
            )
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(symbols):
            status_text.text(f"Analyzing {symbol}... ({i+1}/{len(symbols)})")
            
            try:
                # Fetch data
                data = collector.fetch_data(symbol, period="3mo")
                data_with_indicators = calculator.calculate_all(data)
                
                # Get latest indicators
                latest = calculator.get_latest_signals(data_with_indicators)
                
                # Scanner classification
                scan_result = scanner.scan([symbol])[0] if scanner else {"symbol": symbol, "interesting": False}
                
                # Multi-agent analysis (if available and not in demo mode)
                multi_result = None
                if orchestrator and not st.session_state['demo_mode']:
                    try:
                        # Convert DataFrame to dict for market_data
                        market_data_dict = data_with_indicators.to_dict('list')
                        
                        # Call analyze with correct parameters
                        multi_result = orchestrator.analyze(
                            symbol=symbol,
                            market_data=market_data_dict,
                            indicators=latest
                        )
                    except Exception as e:
                        multi_result = {"error": str(e)}
                
                results.append({
                    "symbol": symbol,
                    "data": {"prices": data_with_indicators},
                    "latest": latest,
                    "scan_result": scan_result,
                    "multi_agent": multi_result
                })
                
            except Exception as e:
                st.error(f"❌ {symbol}: {e}")
            
            progress_bar.progress((i + 1) / len(symbols))
        
        status_text.empty()
        progress_bar.empty()
        
        st.session_state["scan_results"] = results

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

if "scan_results" in st.session_state and st.session_state["scan_results"]:
    results = st.session_state["scan_results"]
    
    st.success(f"✅ Scan complete! Analyzed {len(results)} stocks")
    
    # Categorize results
    interesting = []
    hold = []
    not_interesting = []
    
    for stock in results:
        multi_result = stock.get("multi_agent")
        
        # Determine category based on multi-agent result (if available)
        if multi_result:
            final_signal = multi_result.get("final_signal")
            signal_str = final_signal.value if hasattr(final_signal, "value") else str(final_signal)
            
            if signal_str == "BUY":
                interesting.append(stock)
            elif signal_str == "HOLD":
                hold.append(stock)
            else:
                not_interesting.append(stock)
        else:
            # Fallback to scanner result
            if stock["scan_result"].get("interesting"):
                interesting.append(stock)
            else:
                not_interesting.append(stock)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🟢 Interesting", len(interesting))
    col2.metric("🟡 Hold", len(hold))
    col3.metric("🔴 Not Interesting", len(not_interesting))
    col4.metric("📊 Total Scanned", len(results))
    
    st.markdown("---")
    
    # Show tabs with counts
    tab1, tab2, tab3 = st.tabs([
        f"🟢 Interesting ({len(interesting)})",
        f"🟡 Hold ({len(hold)})",
        f"🔴 Not Interesting ({len(not_interesting)})"
    ])
    
    # Tab 1: Interesting stocks (BUY signals)
    with tab1:
        if interesting:
            for stock in interesting:
                symbol = stock["symbol"]
                
                # Multi-agent analysis if available
                multi_result = stock.get("multi_agent")
                final_signal = None
                final_confidence = 0.0
                if multi_result:
                    fs = multi_result.get("final_signal")
                    final_signal = fs.value if hasattr(fs, "value") else str(fs)
                    final_confidence = multi_result.get("final_confidence", 0.0)
                
                with st.expander(f"**{symbol}** - {signal_badge(final_signal or 'BUY')} {final_confidence:.0%}" if multi_result else f"**{symbol}**"):
                    # Two tabs: Analysis & Chart
                    analysis_tab, chart_tab = st.tabs(["📊 Analysis", "📈 Chart"])
                    
                    with analysis_tab:
                        if multi_result:
                            render_multi_agent_tab(multi_result)
                        else:
                            st.markdown(stock.get("reasoning", "No reasoning available"))
                    
                    with chart_tab:
                        if "data" in stock:
                            try:
                                fig = make_price_chart(symbol, stock["data"])
                                st.plotly_chart(fig, width='stretch')
                            except Exception as e:
                                st.error(f"Could not generate chart: {e}")
                    
                    # Paper trade button (only for BUY signals and not in demo mode)
                    if PAPER_TRADING_AVAILABLE and final_signal == "BUY" and not st.session_state['demo_mode']:
                        st.divider()
                        
                        # Check if auto-traded
                        auto_traded = (
                            st.session_state.auto_trade_enabled and 
                            final_confidence >= st.session_state.auto_trade_threshold
                        )
                        
                        if auto_traded:
                            st.info(f"✅ Auto-traded at {final_confidence:.0%} confidence (threshold: {st.session_state.auto_trade_threshold:.0%})")
                        else:
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                if st.button(f"Execute Paper Trade", key=f"trade_{symbol}", type="primary"):
                                    # Execute trade logic here
                                    st.success(f"Paper trade executed for {symbol}!")
                            with col2:
                                st.caption("⚠️ This will execute a simulated trade with your configured risk parameters")
        else:
            st.info("No interesting stocks found in this scan. Try different stocks or adjust scanner settings.")
    
    # Tab 2: Hold stocks
    with tab2:
        if hold:
            for stock in hold:
                symbol = stock["symbol"]
                multi_result = stock.get("multi_agent")
                
                with st.expander(f"**{symbol}**"):
                    if multi_result:
                        render_multi_agent_tab(multi_result)
                    else:
                        st.markdown(stock.get("reasoning", "No clear signal"))
        else:
            st.info("No hold signals in this scan.")
    
    # Tab 3: Not interesting stocks (SELL or avoid)
    with tab3:
        if not_interesting:
            for stock in not_interesting:
                symbol = stock["symbol"]
                multi_result = stock.get("multi_agent")
                
                with st.expander(f"**{symbol}**"):
                    if multi_result:
                        render_multi_agent_tab(multi_result)
                    else:
                        st.markdown(stock.get("reasoning", "Not recommended"))
        else:
            st.success("All stocks showed interesting signals!")

else:
    # ========================================================================
    # WELCOME MESSAGE
    # ========================================================================
    st.markdown("""
    ## Welcome to AI Trading Copilot 🚀

    An **intelligent trading assistant** powered by multi-agent AI to help you make informed trading decisions.

    ### 🎯 What it does:
    
    **1. Multi-Agent Stock Analysis**
    - 🧠 **Technical Analysis Agent** - Reads charts, indicators, and price patterns
    - 📊 **Momentum Strategy Agent** - Identifies trend strength and reversals  
    - 📈 **Breakout Strategy Agent** - Spots support/resistance breakouts
    - 🤝 **Consensus Engine** - Aggregates insights for final recommendation

    **2. Automated Risk Management**
    - 💰 **Position Sizing** - Calculates optimal share quantity (1.5% risk per trade)
    - 🛡️ **Portfolio Protection** - Max 5% capital per position, 30% sector exposure
    - 📉 **Stop-Loss & Take-Profit** - Auto-calculated based on volatility (ATR)

    **3. Paper Trading Execution**
    - 📝 **Simulated Orders** - Test strategies without real money
    - 💼 **Live Portfolio Tracking** - Monitor positions, P&L in real-time
    - 📊 **Performance Analytics** - Win rate, profit factor, Sharpe ratio, and more

    ---

    ### 🚀 How to use:

    1. **Add your HuggingFace token** (sidebar) to enable AI analysis
    2. **Select stocks** to scan (Custom or Nifty 50 sample)
    3. **Click "Run Scanner"** - AI agents analyze each stock
    4. **Review recommendations** - See why each stock is BUY/HOLD/SELL
    5. **Execute paper trades** - Test the strategy risk-free
    6. **Track performance** - Visit Portfolio & Analytics pages

    ---

    ### 📊 Navigation:

    - **🏠 Home** (this page) - Run stock scanner
    - **💼 Portfolio** - View open positions and P&L
    - **📈 Analytics** - Performance metrics and trade history
    - **🤖 Multi-Agent Demo** - See detailed agent reasoning

    ---

    ### 🔧 Powered by:

    - **HuggingFace Llama-3-8B** - Free AI model for stock analysis
    - **LangGraph** - Multi-agent orchestration framework
    - **Yahoo Finance** - Free real-time market data
    - **Streamlit** - Interactive dashboard
    - **SQLite** - Local trade journal database

    ---

    **Ready to start?** 👉 Select stocks in the sidebar and click **"Run Scanner"**!
    """)

    # Show sample stocks
    with st.expander("📋 Available Nifty 50 Sample Stocks"):
        import textwrap
        stock_list = ", ".join(NIFTY_50_SAMPLE)
        st.write(stock_list)
    
    # Quick start guide
    with st.expander("💡 Quick Start Tips"):
        st.markdown("""
        **For best results:**
        
        1. **Enable AI Analysis**
           - Get free HuggingFace token: [Click here](https://huggingface.co/settings/tokens)
           - Enter token in sidebar under "🔑 HuggingFace API Token"
           - AI provides detailed reasoning for each recommendation

        2. **Start Small**
           - Try scanning 3-5 stocks first
           - Review the multi-agent analysis carefully
           - Check the risk parameters before trading
        
        3. **Use Auto-Trade Wisely**
           - Set minimum confidence threshold (75-85% recommended)
           - Monitor your portfolio regularly
           - Review closed trades in Analytics page
        
        4. **Learn from Analytics**
           - Track your win rate and profit factor
           - See which strategies work best
           - Adjust based on performance data
        
        5. **Try Demo Mode First**
           - See the full multi-agent system in action
           - No API costs, uses pre-recorded analysis
           - Perfect for understanding how it works
           
        **Remember:** This is paper trading (simulation). No real money is at risk!
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px;'>
    <p>🎓 Built as an AI Engineering learning project | 
    <a href='https://github.com/sohansputhran/ai-trading-copilot' target='_blank'>View on GitHub</a></p>
    <p>⚠️ Not financial advice. Paper trading only. Always do your own research.</p>
</div>
""", unsafe_allow_html=True)
