"""
AI Trading Scanner Dashboard

WHY THIS EXISTS:
- Visual interface for the scanner agent
- See scan results in a clean, professional UI
- View charts and analysis

HOW TO RUN:
    streamlit run streamlit_app/app.py
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
from streamlit_app.components.risk_sidebar import render_risk_sidebar

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

    st.markdown("#### Final Decision")
    dec_col1, dec_col2, dec_col3 = st.columns(3)
    with dec_col1:
        if signal_str == "BUY":
            st.success(f"### {signal_badge(signal_str)}")
        elif signal_str == "SELL":
            st.error(f"### {signal_badge(signal_str)}")
        else:
            st.warning(f"### {signal_badge(signal_str)}")
    with dec_col2:
        st.metric("Confidence", f"{final_confidence:.0%}")
    with dec_col3:
        agreement_pct = f"{agent_agreement:.0%}"
        if agent_agreement == 1.0:
            st.metric("Agent Agreement", agreement_pct, "unanimous")
        elif agent_agreement >= 0.67:
            st.metric("Agent Agreement", agreement_pct, "majority")
        else:
            st.metric("Agent Agreement", agreement_pct, "split — low conviction")

    st.markdown("#### Agent Breakdown")
    a_col1, a_col2, a_col3 = st.columns(3)
    tech = multi_result.get("technical_analysis")
    mom  = multi_result.get("momentum_analysis")
    brk  = multi_result.get("breakout_analysis")
    if tech: render_agent_card(tech, a_col1)
    else:    a_col1.error("Technical agent failed")
    if mom:  render_agent_card(mom, a_col2)
    else:    a_col2.error("Momentum agent failed")
    if brk:  render_agent_card(brk, a_col3)
    else:    a_col3.error("Breakout agent failed")

    with st.expander("Full reasoning chain"):
        st.text(final_reasoning)
    if errors:
        with st.expander(f"⚠️ {len(errors)} non-fatal error(s)"):
            for e in errors:
                st.caption(e)

# Paper trading UI helpers

def render_open_positions_table() -> None:
    """Show all open paper positions above the scan results."""
    if not PAPER_TRADING_AVAILABLE:
        return

    import pandas as pd

    order_manager = st.session_state.get("order_manager")
    if order_manager is None:
        return

    open_positions = order_manager.get_open_positions()

    st.subheader("📌 Open Paper Positions")

    if not open_positions:
        st.info("No open positions. Use the **Paper Trade** button on a BUY signal to open one.")
        return

    rows = []
    for o in open_positions:
        rows.append({
            "Symbol":            o.symbol,
            "Shares":            o.shares,
            "Fill Price (₹)":    f"₹{o.fill_price:,.2f}" if o.fill_price else "—",
            "Stop Loss (₹)":     f"₹{o.stop_loss:,.2f}",
            "Take Profit (₹)":   f"₹{o.take_profit:,.2f}",
            "Capital at Risk":   f"₹{o.capital_at_risk:,.0f}",
            "Confidence":        f"{o.confidence:.0%}",
            "Strategy":          o.strategy,
            "Order ID":          o.order_id[:8] + "…",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.caption("Close a position:")
    close_cols = st.columns(min(len(open_positions), 5))
    for idx, o in enumerate(open_positions):
        with close_cols[idx % 5]:
            if st.button(f"Close {o.symbol}", key=f"close_{o.order_id}"):
                with st.spinner(f"Closing {o.symbol}..."):
                    closed = order_manager.close_position(o.order_id, reason="manual")
                if closed and closed.pnl is not None:
                    pnl_color = "green" if closed.pnl >= 0 else "red"
                    st.markdown(
                        f"**{o.symbol}** closed | "
                        f"P&L: <span style='color:{pnl_color}'>₹{closed.pnl:,.2f} "
                        f"({closed.pnl_pct:+.2f}%)</span> | "
                        f"{closed.r_multiple:+.2f}R",
                        unsafe_allow_html=True,
                    )
                st.rerun()

    st.markdown("---")


def render_paper_trade_button(symbol: str, result: dict, position_size, validation) -> None:
    """Render Paper Trade button + auto-trade logic for a single BUY stock card."""
    if not PAPER_TRADING_AVAILABLE:
        return

    order_manager = st.session_state.get("order_manager")
    if order_manager is None:
        return

    # Check if this symbol already has an open position
    open_symbols = {o.symbol for o in order_manager.get_open_positions()}
    if symbol in open_symbols:
        st.info("📌 Position already open for this symbol.")
        return

    indicators = result.get("indicators", {})
    entry_price = result.get("price", 0.0)
    atr = indicators.get("atr", None)
    stop_loss = (
        entry_price - (atr * 1.5)
        if (atr and atr > 0 and entry_price > 0)
        else entry_price * 0.97
    )
    take_profit = entry_price * (1 + 2 * ((entry_price - stop_loss) / entry_price))

    multi_agent_data = result.get("multi_agent") or {}
    final_signal = multi_agent_data.get("final_signal")
    confidence = multi_agent_data.get("final_confidence", 0.0)
    reasoning = multi_agent_data.get("final_reasoning", "")

    decision = RiskDecision(
        symbol=symbol,
        side=OrderSide.BUY,
        shares=position_size.shares,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=confidence,
        strategy="multi_agent",
        agent_reasoning=reasoning,
        capital_at_risk=position_size.capital_at_risk,
    )

    auto_enabled   = st.session_state.get("auto_trade_enabled", False)
    auto_threshold = st.session_state.get("auto_trade_threshold", 1.0)
    auto_key       = f"auto_traded_{symbol}"

    # Auto-trade path — fires once per scan session per symbol
    if auto_enabled and confidence >= auto_threshold and auto_key not in st.session_state:
        order = order_manager.submit(decision)
        st.session_state[auto_key] = order.order_id
        if order.fill_price:
            st.success(
                f"⚡ Auto paper trade executed: {order.shares} × {symbol} "
                f"@ ₹{order.fill_price:,.2f} (slippage ₹{order.slippage:+.2f})"
            )
        else:
            st.error(f"⚡ Auto-trade rejected for {symbol} — could not fetch live price.")
        return

    # Manual button path
    if not validation.approved:
        st.warning("⚠️ Risk check failed — trading anyway at your own discretion.")
    btn_col, status_col = st.columns([1, 3])
    with btn_col:
        clicked = st.button(
            "📋 Paper Trade",
            key=f"paper_trade_{symbol}",
            type="primary",
            disabled=not validation.approved,
            # help=(
            #     "Execute a paper trade at the current live price"
            #     if validation.approved
            #     else "Risk check failed — cannot paper trade"
            # ),
            help="Execute a paper trade at the current live price",
        )
    if clicked:
        with st.spinner(f"Fetching live price for {symbol}..."):
            order = order_manager.submit(decision)
        with status_col:
            if order.fill_price:
                st.success(
                    f"✅ Filled {order.shares} × {symbol} @ ₹{order.fill_price:,.2f} "
                    f"(slippage ₹{order.slippage:+.2f})"
                )
            else:
                st.error(f"❌ Order rejected — could not fetch live price for {symbol}.")


# Session state - initialised once per session

if "portfolio_risk" not in st.session_state:
    st.session_state.portfolio_risk = PortfolioRisk(portfolio_value=PORTFOLIO_VALUE)

if "position_sizer" not in st.session_state:
    st.session_state.position_sizer = PositionSizer(portfolio_value=PORTFOLIO_VALUE)

if "validator" not in st.session_state:
    st.session_state.validator = PreTradeValidator()

# paper trading objects
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

# Page config & title

st.set_page_config(
    page_title="AI Trading Scanner",
    layout="wide"
)

st.title("AI Trading Scanner")
if SCANNER_TYPE == "AI":
    st.markdown("*Powered by free HuggingFace AI*")
else:
    st.markdown("*Using rule-based analysis (no AI needed)*")

# Sidebar

st.sidebar.header("Scanner Settings")

# Risk sidebar — always visible
render_risk_sidebar(
    snapshot=st.session_state.portfolio_risk.snapshot(),
    sizing_method=os.getenv("SIZING_METHOD", "fixed_fractional"),
)

# Stock selection
scan_option = st.sidebar.radio(
    "What to scan:",
    ["Custom Stocks", "Nifty 50 Sample (10 stocks)"]
)

if scan_option == "Custom Stocks":
    stock_input = st.sidebar.text_area(
        "Enter stock symbols (one per line):",
        "RELIANCE.NS\nTCS.NS\nINFY.NS",
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

# Main content

if scan_button:
    if not symbols:
        st.error("Please enter at least one stock symbol!")
    else:
        # Show scanning progress
        st.subheader(f"Scanning {len(symbols)} stocks...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Initialize scanner
        scanner = MarketScanner()

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
                            # A stock is "interesting" if final signal is BUY or SELL
                            final_signal = multi.get("final_signal")
                            if final_signal is not None:
                                signal_value = final_signal.value if hasattr(final_signal, "value") else str(final_signal)
                                result["interesting"] = (signal_value in ("BUY", "SELL"))
                        else:
                            result["multi_agent"] = None
                            
                    except Exception as e:
                        # Log error but don't crash - keep single scanner classification
                        result["multi_agent"] = None
                else:
                    result["multi_agent"] = None
                
                results.append(result)
            progress_bar.progress((i + 1) / len(symbols))
        progress_bar.empty()
        status_text.empty()

        # Separate results into interesting and not interesting
        interesting_stocks = [r for r in results if r.get('interesting', False)]
        not_interesting_stocks = [r for r in results if not r.get('interesting', False)]

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Scanned", len(results))
        with col2:
            st.metric("Interesting", len(interesting_stocks))
        with col3:
            st.metric("Not Interesting", len(not_interesting_stocks))

        # open positions table - shown before scan results
        render_open_positions_table()

        tab1, tab2 = st.tabs(["Interesting Stocks", "Not Interesting"])

        with tab1:
            if interesting_stocks:
                st.subheader(f"{len(interesting_stocks)} Stocks with Clear Signals")

                for i, result in enumerate(interesting_stocks, 1):
                    with st.expander(f"**{i}. {result['symbol']}** - {result['price']:.2f}", expanded=(i==1)):

                        # Create two columns
                        col1, col2 = st.columns([1, 1])

                        with col1:
                            st.markdown("### Indicators")

                            # Show indicators in a nice format
                            indicators = result['indicators']

                            # RSI with color
                            rsi_val = indicators['rsi']
                            if rsi_val < 30:
                                rsi_color, rsi_label = "🟢", "Oversold"
                            elif rsi_val > 70:
                                rsi_color, rsi_label = "🔴", "Overbought"
                            else:
                                rsi_color, rsi_label = "🟡", "Neutral"

                            st.metric("RSI", f"{rsi_val:.2f}", f"{rsi_color} {rsi_label}")

                            # Other indicators
                            st.metric("MACD", f"{indicators['macd']:.2f}")
                            st.metric("Volume Ratio", f"{indicators['volume_ratio']:.2f}x")

                        with col2:
                            st.markdown("### AI Analysis")
                            st.success(result['analysis'].replace('\n', '\n\n'))

                        # Price chart
                        st.markdown("### Price Chart (Last 3 Months)")

                        # Fetch data for chart
                        collector = MarketDataCollector()
                        calc = SimpleTechnicalIndicators()
                        try:
                            price_data = collector.fetch_data(result['symbol'], period="3mo")
                            data_with_ind = calc.calculate_all(price_data)

                            # Create candlestick chart
                            fig = go.Figure()

                            # Add candlestick
                            fig.add_trace(go.Candlestick(
                                x=data_with_ind.index,
                                open=data_with_ind['Open'],
                                high=data_with_ind['High'],
                                low=data_with_ind['Low'],
                                close=data_with_ind['Close'],
                                name='Price'
                            ))

                            # Add Bollinger Bands
                            fig.add_trace(go.Scatter(
                                x=data_with_ind.index,
                                y=data_with_ind['BB_Upper'],
                                name='BB Upper',
                                line=dict(dash='dash', color='gray')
                            ))
                            fig.add_trace(go.Scatter(
                                x=data_with_ind.index,
                                y=data_with_ind['BB_Lower'],
                                name='BB Lower',
                                line=dict(dash='dash', color='gray'),
                                fill='tonexty'
                            ))
                            fig.update_layout(
                                height=400,
                                xaxis_title="Date",
                                yaxis_title="Price (Rupee)",
                                hovermode='x unified'
                            )
                            st.plotly_chart(fig, width="stretch")
                        except Exception as e:
                            st.warning(f"Could not load chart: {str(e)}")

                        # Multi-agent analysis
                        st.markdown("---")
                        st.markdown("### 🤖 Multi-Agent Analysis")
                        render_multi_agent_tab(result.get("multi_agent"))

                        # Risk assessment — BUY signals only
                        multi_agent_data = result.get("multi_agent")
                        final_signal = multi_agent_data.get("final_signal") if multi_agent_data else None
                        final_confidence = multi_agent_data.get("final_confidence", 0.0) if multi_agent_data else 0.0
                        signal_value = final_signal.value if hasattr(final_signal, "value") else str(final_signal)

                        if True:  # Show risk assessment & paper trade for all stocks
                            st.markdown("---")
                            st.markdown("### 🛡️ Risk Assessment")

                            _snap = st.session_state.portfolio_risk.snapshot()
                            _indicators = result.get("indicators", {})
                            _entry = result.get("price", 0.0)
                            _atr = _indicators.get("atr", None)
                            _stop = (
                                _entry - (_atr * 1.5)
                                if (_atr and _atr > 0 and _entry > 0)
                                else _entry * 0.98
                            )

                            _size = st.session_state.position_sizer.calculate(
                                entry_price=_entry,
                                stop_loss=_stop,
                                atr=_atr,
                                confidence=final_confidence,
                                reward_risk_ratio=2.0,
                            )

                            _validation = st.session_state.validator.validate(
                                symbol=result["symbol"],
                                position_value=_size.position_value,
                                portfolio_value=_snap.portfolio_value,
                                open_positions=_snap.open_positions,
                                confidence=final_confidence,
                                daily_pnl=_snap.daily_pnl,
                                sector=None,
                                sector_exposure=0.0,
                                capital_at_risk=_size.capital_at_risk,
                            )

                            r_col1, r_col2, r_col3 = st.columns(3)
                            with r_col1:
                                st.metric("Suggested Shares", _size.shares)
                            with r_col2:
                                st.metric("Capital at Risk", f"{_size.capital_at_risk:,.0f}")
                            with r_col3:
                                st.metric("Position Size", f"{_size.position_value:,.0f}")

                            if _validation.approved:
                                st.success(
                                    f"✅ Risk Check Passed "
                                    f"({_validation.checks_passed}/{_validation.checks_total})"
                                )
                            else:
                                st.error("❌ Risk Check Failed")
                                for _reason in _validation.rejection_reasons:
                                    st.caption(f"• {_reason}")

                            with st.expander("Sizing reasoning", expanded=False):
                                st.caption(_size.reasoning)

                            # Paper Trade button
                            st.markdown("---")
                            st.markdown("### 📋 Paper Trade")
                            render_paper_trade_button(
                                symbol=result["symbol"],
                                result=result,
                                position_size=_size,
                                validation=_validation,
                            )

                            # Update sidebar with proposed trade
                            with st.sidebar:
                                render_risk_sidebar(
                                    snapshot=_snap,
                                    sizing_method=os.getenv("SIZING_METHOD", "fixed_fractional"),
                                    selected_symbol=result["symbol"],
                                    proposed_size={
                                        "shares": _size.shares,
                                        "position_value": _size.position_value,
                                        "capital_at_risk": _size.capital_at_risk,
                                        "fraction_used": _size.fraction_used,
                                        "reasoning": _size.reasoning,
                                        "approved": _validation.approved,
                                        "rejection_reasons": _validation.rejection_reasons,
                                    },
                                )

                                # Analytics page link
                                st.sidebar.divider()
                                st.sidebar.markdown("### 📊 Analytics")
                                st.sidebar.page_link(
                                    "pages/4_analytics.py",
                                    label="View Trade Analytics",
                                    icon="📈",
                                )

            else:
                st.info("No stocks with clear signals found.")

        with tab2:
            if not_interesting_stocks:
                st.subheader(f"{len(not_interesting_stocks)} Stocks Without Clear Signals")
                st.caption("These stocks don't show strong technical patterns right now")

                for i, result in enumerate(not_interesting_stocks, 1):
                    with st.expander(f"**{i}. {result['symbol']}** - {result['price']:.2f}"):

                        # Create two columns
                        col1, col2 = st.columns([1, 1])

                        with col1:
                            st.markdown("### Indicators")
                            indicators = result['indicators']

                            # RSI with color
                            rsi_val = indicators['rsi']
                            if rsi_val < 30:
                                rsi_color, rsi_label = "🟢", "Oversold"
                            elif rsi_val > 70:
                                rsi_color, rsi_label = "🔴", "Overbought"
                            else:
                                rsi_color, rsi_label = "🟡", "Neutral"

                            st.metric("RSI", f"{rsi_val:.2f}", f"{rsi_color} {rsi_label}")
                            st.metric("MACD", f"{indicators['macd']:.2f}")
                            st.metric("Volume Ratio", f"{indicators['volume_ratio']:.2f}x")

                        with col2:
                            st.markdown("### AI Analysis")
                            st.info(result['analysis'].replace('\n', '\n\n'))

                        # Multi-agent analysis
                        st.markdown("---")
                        st.markdown("### 🤖 Multi-Agent Analysis")
                        render_multi_agent_tab(result.get("multi_agent"))

                        # Risk assessment & Paper Trade for not-interesting stocks too
                        multi_agent_data = result.get("multi_agent")
                        final_confidence = multi_agent_data.get("final_confidence", 0.0) if multi_agent_data else 0.0

                        st.markdown("---")
                        st.markdown("### 🛡️ Risk Assessment")

                        _snap = st.session_state.portfolio_risk.snapshot()
                        _indicators = result.get("indicators", {})
                        _entry = result.get("price", 0.0)
                        _atr = _indicators.get("atr", None)
                        _stop = (
                            _entry - (_atr * 1.5)
                            if (_atr and _atr > 0 and _entry > 0)
                            else _entry * 0.98
                        )

                        _size = st.session_state.position_sizer.calculate(
                            entry_price=_entry,
                            stop_loss=_stop,
                            atr=_atr,
                            confidence=final_confidence,
                            reward_risk_ratio=2.0,
                        )

                        _validation = st.session_state.validator.validate(
                            symbol=result["symbol"],
                            position_value=_size.position_value,
                            portfolio_value=_snap.portfolio_value,
                            open_positions=_snap.open_positions,
                            confidence=final_confidence,
                            daily_pnl=_snap.daily_pnl,
                            sector=None,
                            sector_exposure=0.0,
                            capital_at_risk=_size.capital_at_risk,
                        )

                        r_col1, r_col2, r_col3 = st.columns(3)
                        with r_col1:
                            st.metric("Suggested Shares", _size.shares)
                        with r_col2:
                            st.metric("Capital at Risk", f"{_size.capital_at_risk:,.0f}")
                        with r_col3:
                            st.metric("Position Size", f"{_size.position_value:,.0f}")

                        if _validation.approved:
                            st.success(
                                f"✅ Risk Check Passed "
                                f"({_validation.checks_passed}/{_validation.checks_total})"
                            )
                        else:
                            st.error("❌ Risk Check Failed")
                            for _reason in _validation.rejection_reasons:
                                st.caption(f"• {_reason}")

                        # # Paper Trade button
                        # st.markdown("---")
                        # st.markdown("### 📋 Paper Trade")
                        # render_paper_trade_button(
                        #     symbol=result["symbol"],
                        #     result=result,
                        #     position_size=_size,
                        #     validation=_validation,
                        # )

            else:
                st.success("All stocks showed interesting signals!")

else:
    # Welcome message
    st.markdown("""
    ## Welcome to AI Trading Scanner

    This tool uses **free AI** to scan stocks and identify potential trading opportunities.

    ### How it works:
    1. **Select stocks** to scan (left sidebar)
    2. **Click "Run Scanner"** to start
    3. **View results** - AI will explain why each stock is interesting

    ### What the AI looks for:
    - 🟢 **Oversold stocks** (RSI < 30) - potential buy opportunities
    - 🔴 **Overbought stocks** (RSI > 70) - potential sell opportunities
    - 📊 **Momentum shifts** (MACD changes)
    - 📈 **Volume spikes** (unusual trading activity)

    ### Powered by:
    - **HuggingFace Llama-3-8B** (Free AI model)
    - **LangGraph** (Multi-agent orchestration)
    - **Yahoo Finance** (Free market data)
    - **Streamlit** (This beautiful dashboard)

    ---

    **Ready?** Select stocks in the sidebar and click "Run Scanner"!
    """)

    # Show sample stocks
    with st.expander("Available Nifty 50 Sample Stocks"):
        st.write(", ".join(NIFTY_50_SAMPLE))

# Footer
st.markdown("---")
st.markdown("*Built as an AI Engineering learning project. Not financial advice.*")