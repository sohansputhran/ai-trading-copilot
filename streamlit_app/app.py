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

# ─────────────────────────────────────────────
# Risk engine - initialised once per session
# ─────────────────────────────────────────────

if "portfolio_risk" not in st.session_state:
    st.session_state.portfolio_risk = PortfolioRisk(portfolio_value=PORTFOLIO_VALUE)

if "position_sizer" not in st.session_state:
    st.session_state.position_sizer = PositionSizer(portfolio_value=PORTFOLIO_VALUE)

if "validator" not in st.session_state:
    st.session_state.validator = PreTradeValidator()

# Page config
st.set_page_config(
    page_title="AI Trading Scanner",
    layout="wide"
)

# Title
st.title("AI Trading Scanner")
if SCANNER_TYPE == "AI":
    st.markdown("*Powered by free HuggingFace AI*")
else:
    st.markdown("*Using rule-based analysis (no AI needed)*")

# Sidebar - Input
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
        for i, symbol in enumerate(symbols):
            status_text.text(f"Scanning {symbol}... ({i+1}/{len(symbols)})")
            result = scanner.scan_stock(symbol)
            if result:
                # Run multi-agent analysis and attach to result
                if orchestrator is not None:
                    try:
                        multi = orchestrator.analyze(
                            symbol,
                            result.get("indicators", {}),  # market_data
                            result.get("indicators", {}),  # indicators (same dict)
                        )
                        result["multi_agent"] = multi

                        # Override single scanner classification with multi-agent decision.
                        # A stock is "interesting" if the aggregator produced a
                        # non-HOLD signal - confidence threshold already enforced
                        # inside aggregator.py, so we trust the output directly.
                        final_signal = multi.get("final_signal")
                        if final_signal is not None:
                            result["interesting"] = (
                                final_signal.value in ("BUY", "SELL")
                            )
                    except Exception as e:
                        result["multi_agent"] = None
                        # Keep single scanner classification on failure
                else:
                    result["multi_agent"] = None
                    # No orchestrator - single scanner classification stands
                results.append(result)
            progress_bar.progress((i + 1) / len(symbols))

        progress_bar.empty()
        status_text.empty()

        # Separate results into interesting and not interesting
        interesting_stocks = [r for r in results if r.get('interesting', False)]
        not_interesting_stocks = [r for r in results if not r.get('interesting', False)]

        # Display summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Scanned", len(results))
        with col2:
            st.metric("Interesting", len(interesting_stocks))
        with col3:
            st.metric("Not Interesting", len(not_interesting_stocks))

        # Create tabs for interesting vs not interesting
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
                                rsi_color = "🟢"  # Oversold
                                rsi_label = "Oversold"
                            elif rsi_val > 70:
                                rsi_color = "🔴"  # Overbought
                                rsi_label = "Overbought"
                            else:
                                rsi_color = "🟡"  # Neutral
                                rsi_label = "Neutral"

                            st.metric("RSI", f"{rsi_val:.2f}", f"{rsi_color} {rsi_label}")

                            # Other indicators
                            st.metric("MACD", f"{indicators['macd']:.2f}")
                            st.metric("Volume Ratio", f"{indicators['volume_ratio']:.2f}x")

                        with col2:
                            st.markdown("### AI Analysis")
                            st.success(result['analysis'].replace('\n', '\n\n'))

                        # Add a chart
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

                        # Multi-agent analysis tab
                        st.markdown("---")
                        st.markdown("### 🤖 Multi-Agent Analysis")
                        render_multi_agent_tab(result.get("multi_agent"))

                        # Risk assessment — only for BUY signals
                        multi_agent_data = result.get("multi_agent")
                        final_signal = multi_agent_data.get("final_signal") if multi_agent_data else None
                        final_confidence = multi_agent_data.get("final_confidence", 0.0) if multi_agent_data else 0.0
                        signal_value = final_signal.value if hasattr(final_signal, "value") else str(final_signal)

                        if signal_value == "BUY":
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

                            # Update sidebar with this stock's proposed trade
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
                                rsi_color = "🟢"
                                rsi_label = "Oversold"
                            elif rsi_val > 70:
                                rsi_color = "🔴"
                                rsi_label = "Overbought"
                            else:
                                rsi_color = "🟡"
                                rsi_label = "Neutral"

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
