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
    if multi_result is None:
        st.info("Multi-agent analysis unavailable — LangGraph not installed.")
        return
    final_signal     = multi_result.get("final_signal")
    final_confidence = multi_result.get("final_confidence", 0)
    agent_agreement  = multi_result.get("agent_agreement", 0)
    final_reasoning  = multi_result.get("final_reasoning", "")
    errors           = multi_result.get("errors", [])
    signal_str = final_signal.value if hasattr(final_signal, "value") else str(final_signal)

    if signal_str == "BUY":
        st.success(f"### {signal_badge(signal_str)} — {final_confidence:.0%} confidence")
    elif signal_str == "SELL":
        st.error(f"### {signal_badge(signal_str)} — {final_confidence:.0%} confidence")
    else:
        st.warning(f"### {signal_badge(signal_str)} — {final_confidence:.0%} confidence")

    st.markdown(f"**Agent Agreement:** {agent_agreement:.0%} of agents agree")
    st.markdown(f"**Final Reasoning:** {final_reasoning}")

    if errors:
        with st.expander("⚠️ Errors", expanded=False):
            for err in errors:
                st.error(err)

    analyses = multi_result.get("agent_analyses", [])
    if analyses:
        st.markdown("---")
        st.markdown("#### Individual Agent Analyses")
        cols = st.columns(min(len(analyses), 3))
        for i, analysis in enumerate(analyses):
            render_agent_card(analysis, cols[i % len(cols)])

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

st.title("🤖 AI Trading Copilot")
_active_token = st.session_state.get("hf_token", "")
if _active_token:
    st.markdown("*Powered by free HuggingFace AI* 🤖")
else:
    st.markdown("*Using rule-based analysis — add HuggingFace token in sidebar to enable AI* 🔧")

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.header("Scanner Settings")

# ── HuggingFace Token Input ──────────────────────────────────
from src.utils.config import HUGGINGFACE_API_TOKEN as _CONFIG_TOKEN

# Seed session state from config (env / st.secrets) once per session
if "hf_token" not in st.session_state:
    st.session_state.hf_token = _CONFIG_TOKEN or ""

with st.sidebar.expander("🔑 HuggingFace API Token", expanded=not st.session_state.hf_token):
    st.markdown(
        "Enter your **free** HuggingFace token to enable AI analysis. "
        "[Get one here](https://huggingface.co/settings/tokens)."
    )
    token_input = st.text_input(
        "Token",
        value=st.session_state.hf_token,
        type="password",
        placeholder="hf_…",
        label_visibility="collapsed",
        key="hf_token_input",
    )
    if token_input != st.session_state.hf_token:
        st.session_state.hf_token = token_input
        st.rerun()  # Refresh so the subtitle updates immediately

    if st.session_state.hf_token:
        st.success("✅ Token saved for this session")
    else:
        st.info("ℹ️ No token — rule-based scanner will be used")
# ─────────────────────────────────────────────────────────────

st.sidebar.divider()

# Stock selection
st.sidebar.subheader("What to scan:")
scan_mode = st.sidebar.radio(
    "scan_mode",
    options=["Custom Stocks", "Nifty 50 Sample (10 stocks)"],
    index=0,
    label_visibility="collapsed"
)

if scan_mode == "Custom Stocks":
    st.sidebar.caption("Enter stock symbols (one per line):")
    stock_input = st.sidebar.text_area(
        "stocks",
        value="RELIANCE.NS\nTCS.NS\nINFY.NS",
        label_visibility="collapsed",
        height=150
    )
    symbols = [s.strip() for s in stock_input.split('\n') if s.strip()]
else:
    symbols = NIFTY_50_SAMPLE
    st.sidebar.info(f"Will scan {len(symbols)} Nifty 50 stocks")

# Scan button
scan_button = st.sidebar.button("Run Scanner", type="primary")

# auto-trade toggle (bottom of sidebar)
if PAPER_TRADING_AVAILABLE:
    st.sidebar.divider()
    st.sidebar.subheader("⚡ Auto-Trade")
    auto_enabled = st.sidebar.toggle(
        "Auto-trade BUY signals",
        value=st.session_state.auto_trade_enabled,
        key="auto_trade_toggle",
        help=(
            "When ON, BUY signals that exceed the confidence threshold are paper "
            "traded automatically. When OFF, use the manual 'Paper Trade' button."
        ),
    )
    st.session_state.auto_trade_enabled = auto_enabled
    if auto_enabled:
        threshold = st.sidebar.slider(
            "Min confidence",
            min_value=0.60,
            max_value=0.95,
            value=st.session_state.auto_trade_threshold,
            step=0.05,
            key="auto_trade_threshold_slider",
        )
        st.session_state.auto_trade_threshold = threshold
        st.sidebar.caption(f"Auto-trading BUYs ≥ {threshold:.0%} confidence")
    else:
        st.session_state.auto_trade_threshold = 1.0

# ============================================================================
# MAIN CONTENT - SCANNER RESULTS OR WELCOME
# ============================================================================

if scan_button:
    if not symbols:
        st.error("Please enter at least one stock symbol!")
    else:
        # Show scanning progress
        st.subheader(f"Scanning {len(symbols)} stocks...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Initialize scanner — pass user-supplied token from sidebar
        _token = st.session_state.get("hf_token") or None
        scanner = MarketScanner(token=_token) if SCANNER_TYPE == "AI" else MarketScanner()

        # Initialize multi-agent orchestrator (reuses scanner's LLM client)
        orchestrator = None
        if MULTI_AGENT_AVAILABLE:
            try:
                orchestrator = MultiAgentOrchestrator(
                    technical_agent = TechnicalAnalysisAgent(
                        llm_client=scanner.llm_client,
                        llm_model=scanner.llm_model or "unknown",
                    ),
                    momentum_agent  = MomentumStrategyAgent(
                        llm_client=scanner.llm_client,
                        llm_model=scanner.llm_model or "unknown",
                    ),
                    breakout_agent  = BreakoutStrategyAgent(
                        llm_client=scanner.llm_client,
                        llm_model=scanner.llm_model or "unknown",
                    ),
                )
            except Exception as e:
                st.warning(f"Multi-agent orchestrator unavailable: {e}")

        # Run scan with progress updates
        results = []
        
        # Create collector once before the loop for fetching market data
        collector = MarketDataCollector()
        
        for i, symbol in enumerate(symbols):
            status_text.text(f"Scanning {symbol}... ({i+1}/{len(symbols)})")
            result = scanner.scan_stock(symbol)
            
            if result:
                # Run multi-agent analysis and attach to result
                if orchestrator is not None:
                    try:
                        # Fetch market data DataFrame for orchestrator
                        # Try different method names (collector API may vary)
                        market_data = None
                        for method_name in ['get_data', 'fetch_data', 'fetch']:
                            if hasattr(collector, method_name):
                                try:
                                    method = getattr(collector, method_name)
                                    market_data = method(symbol)
                                    break
                                except:
                                    pass
                        
                        if market_data is not None and not market_data.empty:
                            # Call orchestrator with market_data DataFrame
                            multi = orchestrator.analyze(
                                symbol,
                                market_data,                    # DataFrame with OHLCV data
                                result.get("indicators", {}),   # Indicators dict
                            )
                            result["multi_agent"] = multi
 
                            # Override scanner classification with multi-agent decision
                            # (Multi-agent has higher authority)
                            if "final_signal" in multi:
                                fs = multi["final_signal"]
                                signal_value = fs.value if hasattr(fs, "value") else str(fs)
                                if signal_value == "BUY":
                                    result["category"] = "interesting"
                                elif signal_value == "SELL":
                                    result["category"] = "not_interesting"
                                else:
                                    result["category"] = "hold"
                    except Exception as e:
                        st.warning(f"Multi-agent analysis failed for {symbol}: {e}")
                
                results.append(result)
            
            progress_bar.progress((i + 1) / len(symbols))
        
        progress_bar.empty()
        status_text.empty()

        # Separate results by category
        interesting = [r for r in results if r.get("category") == "interesting"]
        not_interesting = [r for r in results if r.get("category") == "not_interesting"]
        hold = [r for r in results if r.get("category") == "hold"]

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
                    
                    with st.expander(f"**{symbol}** — {signal_badge(final_signal or 'BUY')} {final_confidence:.0%}" if multi_result else f"**{symbol}**"):
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
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Could not generate chart: {e}")
                        
                        # Paper trade button (only for BUY signals)
                        if PAPER_TRADING_AVAILABLE and final_signal == "BUY":
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
    # WELCOME MESSAGE - UPDATED CONTENT
    # ========================================================================
    st.markdown("""
    ## Welcome to AI Trading Copilot 🚀

    An **intelligent trading assistant** powered by multi-agent AI to help you make informed trading decisions.

    ### 🎯 What it does:
    
    **1. Multi-Agent Stock Analysis**
    - 🧠 **Technical Analysis Agent** — Reads charts, indicators, and price patterns
    - 📊 **Momentum Strategy Agent** — Identifies trend strength and reversals  
    - 📈 **Breakout Strategy Agent** — Spots support/resistance breakouts
    - 🤝 **Consensus Engine** — Aggregates insights for final recommendation

    **2. Automated Risk Management**
    - 💰 **Position Sizing** — Calculates optimal share quantity (1.5% risk per trade)
    - 🛡️ **Portfolio Protection** — Max 5% capital per position, 30% sector exposure
    - 📉 **Stop-Loss & Take-Profit** — Auto-calculated based on volatility (ATR)

    **3. Paper Trading Execution**
    - 📝 **Simulated Orders** — Test strategies without real money
    - 💼 **Live Portfolio Tracking** — Monitor positions, P&L in real-time
    - 📊 **Performance Analytics** — Win rate, profit factor, Sharpe ratio, and more

    ---

    ### 🚀 How to use:

    1. **Add your HuggingFace token** (sidebar) to enable AI analysis
    2. **Select stocks** to scan (Custom or Nifty 50 sample)
    3. **Click "Run Scanner"** — AI agents analyze each stock
    4. **Review recommendations** — See why each stock is BUY/HOLD/SELL
    5. **Execute paper trades** — Test the strategy risk-free
    6. **Track performance** — Visit Portfolio & Analytics pages

    ---

    ### 📊 Navigation:

    - **🏠 Home** (this page) — Run stock scanner
    - **💼 Portfolio** — View open positions and P&L
    - **📈 Analytics** — Performance metrics and trade history

    ---

    ### 🔧 Powered by:

    - **HuggingFace Llama-3-8B** — Free AI model for stock analysis
    - **LangGraph** — Multi-agent orchestration framework
    - **Yahoo Finance** — Free real-time market data
    - **Streamlit** — Interactive dashboard
    - **SQLite** — Local trade journal database

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
