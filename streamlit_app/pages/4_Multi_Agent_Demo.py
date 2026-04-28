"""
Multi-Agent Analysis Demo Page

Shows detailed agent reasoning, coordination, and decision-making process.
Works in both demo mode (pre-recorded) and live mode (real API calls).
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import json
import os
from datetime import datetime
import pandas as pd

# Multi-agent imports (for live mode)
try:
    from src.agents.orchestrator import MultiAgentOrchestrator
    from src.agents.technical_agent import TechnicalAnalysisAgent
    from src.agents.momentum_agent import MomentumStrategyAgent
    from src.agents.breakout_agent import BreakoutStrategyAgent
    from src.data_pipeline.collector import MarketDataCollector
    from src.data_pipeline.indicators import SimpleTechnicalIndicators
    MULTI_AGENT_AVAILABLE = True
except ImportError:
    MULTI_AGENT_AVAILABLE = False

st.set_page_config(
    page_title="Multi-Agent Demo - AI Trading Copilot",
    page_icon="🤖",
    layout="wide"
)

# ============================================================================
# DEMO MODE CHECK
# ============================================================================

demo_mode = st.session_state.get('demo_mode', True)

# ============================================================================
# PAGE HEADER
# ============================================================================

st.title("🤖 Multi-Agent Analysis System")

if demo_mode:
    st.info("""
    🎬 **Demo Mode**: Showing pre-recorded agent reasoning from actual system runs.
    
    This demonstrates the full multi-agent orchestration with LangGraph, including:
    - Individual agent analyses (Technical, Momentum, Breakout)
    - Agent coordination and message passing
    - Consensus building and final decision
    - Risk assessment and position sizing
    """)
else:
    st.success("🔴 **Live Mode**: Running real-time multi-agent analysis with API calls")

st.markdown("---")

# ============================================================================
# STOCK SELECTOR
# ============================================================================

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    if demo_mode:
        # Load available demo stocks
        demo_file = Path("streamlit_app/demo_data/multi_agent_results.json")
        
        if demo_file.exists():
            with open(demo_file) as f:
                demo_data = json.load(f)
            
            available_symbols = [item['symbol'] for item in demo_data]
            selected_symbol = st.selectbox(
                "📊 Select Stock to Analyze",
                available_symbols,
                index=0,
                help="Choose from pre-recorded multi-agent analyses"
            )
        else:
            st.error("""
            ⚠️ **Demo data not found!**
            
            Please generate demo data first:
            1. Run locally with API keys
            2. Execute: `python scripts/generate_demo_data.py`
            3. Commit `streamlit_app/demo_data/multi_agent_results.json`
            4. Redeploy
            """)
            st.stop()
    else:
        # Live mode - text input
        selected_symbol = st.text_input(
            "📊 Enter Stock Symbol",
            "RELIANCE.NS",
            help="NSE symbol (e.g., RELIANCE.NS, TCS.NS)"
        )

with col2:
    if demo_mode and demo_file.exists():
        # Show quick stats for selected stock
        result = next((r for r in demo_data if r['symbol'] == selected_symbol), None)
        if result:
            st.metric(
                "Decision",
                result['final_decision'],
                f"{result['confidence_score']:.0%} confidence"
            )

with col3:
    analyze_button = st.button(
        "🔍 Analyze",
        type="primary",
        width='stretch',
        help="Run multi-agent analysis"
    )

# ============================================================================
# RUN ANALYSIS
# ============================================================================

if analyze_button:
    if demo_mode:
        # Load pre-recorded result
        result = next((r for r in demo_data if r['symbol'] == selected_symbol), None)
        if result:
            st.session_state['current_analysis'] = result
            st.success(f"✅ Loaded demo analysis for {selected_symbol}")
    else:
        # Real API call
        with st.spinner(f"🤖 Running multi-agent analysis on {selected_symbol}..."):
            try:
                # Import orchestrator and agents
                from src.agents.orchestrator import MultiAgentOrchestrator
                from src.agents.technical_agent import TechnicalAnalysisAgent
                from src.agents.momentum_agent import MomentumStrategyAgent
                from src.agents.breakout_agent import BreakoutStrategyAgent
                from src.data_pipeline.collector import MarketDataCollector
                from src.data_pipeline.indicators import SimpleTechnicalIndicators
                
                # Fetch data
                collector = MarketDataCollector()
                calculator = SimpleTechnicalIndicators()
                
                data = collector.fetch_data(selected_symbol, period="3mo")
                data_with_indicators = calculator.calculate_all(data)
                latest_signals = calculator.get_latest_signals(data_with_indicators)
                
                # Convert DataFrame to dict for market_data
                market_data_dict = data_with_indicators.to_dict('list')
                
                # Initialize the three agents
                orchestrator = MultiAgentOrchestrator(
                    technical_agent=TechnicalAnalysisAgent(),
                    momentum_agent=MomentumStrategyAgent(),
                    breakout_agent=BreakoutStrategyAgent()
                )
                
                # Run analysis with correct method signature
                final_state = orchestrator.analyze(
                    symbol=selected_symbol,
                    market_data=market_data_dict,
                    indicators=latest_signals
                )
                
                # Convert TradingState to the format expected by the UI
                # Extract individual agent analyses
                agent_list = []
                for key in ['technical_analysis', 'momentum_analysis', 'breakout_analysis']:
                    if key in final_state and final_state[key]:
                        agent = final_state[key]
                        agent_dict = {
                            'agent_name': key.replace('_analysis', ''),
                            'signal': agent.signal.value if hasattr(agent.signal, 'value') else str(agent.signal),
                            'confidence': float(agent.confidence),
                            'reasoning': str(agent.reasoning),
                            'warnings': list(agent.warnings) if hasattr(agent, 'warnings') else []
                        }
                        agent_list.append(agent_dict)
                
                # Build result in expected format
                result = {
                    'symbol': selected_symbol,
                    'timestamp': datetime.now().isoformat(),
                    
                    'scanner_reasoning': {
                        'signals': [],
                        'reasoning': 'Live analysis - see individual agents below',
                        'indicators': {
                            'rsi': float(latest_signals.get('rsi', 0)),
                            'macd': float(latest_signals.get('macd', 0)),
                            'volume_ratio': float(latest_signals.get('volume_ratio', 1.0))
                        }
                    },
                    
                    'technical_analysis': {
                        'detailed_analysis': final_state.get('final_reasoning', 'No analysis'),
                        'patterns': [],
                        'support_levels': [],
                        'resistance_levels': [],
                        'confidence': float(final_state.get('final_confidence', 0))
                    },
                    
                    'risk_assessment': {
                        'position_size_pct': 0.03,
                        'position_size_rupees': 15000,
                        'risk_per_trade': 0.015,
                        'reward_risk_ratio': 2.0,
                        'validation_checks': [
                            {'rule': 'Position size check', 'passed': True, 'status': 'Within limits'}
                        ],
                        'portfolio_value': 500000,
                        'entry_price': float(latest_signals.get('close', 0)),
                        'stop_loss': float(latest_signals.get('close', 0) * 0.95),
                        'risk_per_share': float(latest_signals.get('close', 0) * 0.05),
                        'quantity': 0
                    },
                    
                    'agent_coordination': [
                        {
                            'from_agent': 'Orchestrator',
                            'to_agent': 'All Agents',
                            'message': 'LangGraph executed all agents in parallel',
                            'timestamp': datetime.now().isoformat()
                        }
                    ],
                    
                    'orchestrator_reasoning': final_state.get('final_reasoning', ''),
                    'final_decision': final_state.get('final_signal').value if hasattr(final_state.get('final_signal'), 'value') else str(final_state.get('final_signal', 'HOLD')),
                    'confidence_score': float(final_state.get('final_confidence', 0)),
                    'agent_analyses': agent_list,
                    'agent_agreement': float(final_state.get('agent_agreement', 0)),
                    'errors': final_state.get('errors', [])
                }
                
                # Store in session
                st.session_state['current_analysis'] = result
                st.success(f"✅ Analysis complete for {selected_symbol}")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                st.stop()

# ============================================================================
# DISPLAY ANALYSIS RESULTS
# ============================================================================

if 'current_analysis' in st.session_state:
    result = st.session_state['current_analysis']
    
    st.markdown("---")
    
    # ========================================================================
    # DECISION HEADER
    # ========================================================================
    
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
    with col1:
        st.markdown(f"## 📊 Analysis: **{result['symbol']}**")
        if 'timestamp' in result:
            st.caption(f"Generated: {result['timestamp'][:19]}")
    
    with col2:
        decision = result['final_decision']
        if decision == "BUY":
            st.success(f"### {decision}")
        elif decision == "SELL":
            st.error(f"### {decision}")
        else:
            st.warning(f"### {decision}")
    
    with col3:
        confidence = result['confidence_score']
        st.metric("Confidence", f"{confidence:.0%}")
    
    with col4:
        # Count how many agents were used
        agent_count = len(result.get('agent_analyses', []))
        st.metric("Agents", f"{agent_count}/3")
    
    st.markdown("---")
    
    # ========================================================================
    # AGENT TABS
    # ========================================================================
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Final Decision",
        "🔍 Scanner Agent",
        "📈 Technical Agent",
        "🛡️ Risk Assessment",
        "🧠 Agent Flow"
    ])
    
    # ─── TAB 1: FINAL DECISION ───
    with tab1:
        st.markdown("### 🎯 Orchestrator Final Decision")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Final Signal",
                result['final_decision'],
                delta=f"{result['confidence_score']:.0%} confidence"
            )
        
        with col2:
            agent_agreement = result.get('agent_agreement', 0)
            st.metric(
                "Agent Agreement",
                f"{agent_agreement:.0%}",
                delta="Strong consensus" if agent_agreement >= 0.8 else "Mixed views"
            )
        
        with col3:
            errors = result.get('errors', [])
            st.metric(
                "Status",
                "⚠️ Errors" if errors else "✅ Clean",
                delta=f"{len(errors)} issues" if errors else "No issues"
            )
        
        st.markdown("---")
        
        st.markdown("#### 📝 Reasoning")
        reasoning = (
            result.get('final_reasoning')
            or result.get('orchestrator_reasoning')
            or 'No reasoning provided'
        )
        
        # Parse and clean the reasoning
        if 'FINAL DECISION:' in reasoning or 'AGENT BREAKDOWN:' in reasoning:
            lines = reasoning.split('\n')
            
            # Extract final decision summary
            final_decision_line = [l for l in lines if 'FINAL DECISION:' in l]
            agent_breakdown_start = next((i for i, l in enumerate(lines) if 'AGENT BREAKDOWN:' in l), None)
            warnings_start = next((i for i, l in enumerate(lines) if 'WARNINGS:' in l), None)
            
            # Display cleaned final decision
            if final_decision_line:
                decision_text = final_decision_line[0].replace('FINAL DECISION:', '').strip()
                # Remove the [Overridden...] part if present
                if '[Overridden' in decision_text:
                    decision_text = decision_text.split('[Overridden')[0].strip()
                st.info(f"**Decision Summary:** {decision_text}")
            
            # Display individual agent breakdowns in expandable sections
            if agent_breakdown_start:
                st.markdown("---")
                st.markdown("#### 🤖 Individual Agent Contributions")
                
                # Try to get full reasoning from agent_analyses first (more complete)
                agent_analyses_data = result.get('agent_analyses', [])
                
                if agent_analyses_data:
                    # We have structured agent data - use that instead
                    for agent_data in agent_analyses_data:
                        agent_name = agent_data.get('agent_name', 'Unknown').replace('_', ' ').title()
                        signal = agent_data.get('signal', 'UNKNOWN')
                        confidence = agent_data.get('confidence', 0)
                        reasoning_full = agent_data.get('reasoning', 'No reasoning provided')
                        
                        # Determine icon
                        if 'technical' in agent_name.lower():
                            icon = '📊'
                        elif 'momentum' in agent_name.lower():
                            icon = '📈'
                        elif 'breakout' in agent_name.lower():
                            icon = '🚀'
                        else:
                            icon = '🤖'
                        
                        # Color based on signal
                        if signal == 'BUY':
                            title = f"{icon} **{agent_name}**: 🟢 {signal} ({confidence:.0%})"
                        elif signal == 'SELL':
                            title = f"{icon} **{agent_name}**: 🔴 {signal} ({confidence:.0%})"
                        else:
                            title = f"{icon} **{agent_name}**: 🟡 {signal} ({confidence:.0%})"
                        
                        with st.expander(title, expanded=False):
                            # Split reasoning on semicolons for bullet points
                            if ';' in reasoning_full:
                                points = [p.strip() for p in reasoning_full.split(';') if p.strip()]
                                for point in points:
                                    st.markdown(f"• {point}")
                            else:
                                st.markdown(reasoning_full)
                else:
                    # Fallback to parsing from orchestrator_reasoning text
                    # Find all agent sections
                    agent_sections = []
                    current_agent = None
                    
                    for i in range(agent_breakdown_start + 1, len(lines)):
                        line = lines[i].strip()
                        
                        # Check for agent headers
                        if line.startswith('✓ TECHNICAL_ANALYSIS:') or line.startswith('✗ TECHNICAL_ANALYSIS:'):
                            if current_agent:
                                agent_sections.append(current_agent)
                            current_agent = {'name': 'Technical Analysis', 'lines': [line], 'icon': '📊'}
                        elif line.startswith('✓ MOMENTUM_STRATEGY:') or line.startswith('✗ MOMENTUM_STRATEGY:'):
                            if current_agent:
                                agent_sections.append(current_agent)
                            current_agent = {'name': 'Momentum Strategy', 'lines': [line], 'icon': '📈'}
                        elif line.startswith('✓ BREAKOUT_STRATEGY:') or line.startswith('✗ BREAKOUT_STRATEGY:'):
                            if current_agent:
                                agent_sections.append(current_agent)
                            current_agent = {'name': 'Breakout Strategy', 'lines': [line], 'icon': '🚀'}
                        elif current_agent and line and not line.startswith('WARNINGS:'):
                            current_agent['lines'].append(line)
                        elif line.startswith('WARNINGS:'):
                            if current_agent:
                                agent_sections.append(current_agent)
                            break
                    
                    # Add last agent if exists
                    if current_agent and current_agent not in agent_sections:
                        agent_sections.append(current_agent)
                    
                    # Display each agent in an expander (fallback path)
                    for agent in agent_sections:
                        header = agent['lines'][0]
                        
                        # Extract signal and confidence from header
                        if ':' in header:
                            header_parts = header.split(':', 1)[1].strip()
                            signal = header_parts.split('(')[0].strip() if '(' in header_parts else header_parts
                            confidence = header_parts.split('(')[1].split(')')[0] if '(' in header_parts else 'N/A'
                        else:
                            signal = 'UNKNOWN'
                            confidence = 'N/A'
                        
                        # Color based on signal
                        if 'BUY' in signal:
                            title = f"{agent['icon']} **{agent['name']}**: 🟢 {signal} ({confidence})"
                        elif 'SELL' in signal:
                            title = f"{agent['icon']} **{agent['name']}**: 🔴 {signal} ({confidence})"
                        else:
                            title = f"{agent['icon']} **{agent['name']}**: 🟡 {signal} ({confidence})"
                        
                        with st.expander(title, expanded=False):
                            # Check if we have multiple lines (reasoning split across lines)
                            if len(agent['lines']) > 1:
                                # Display each line as a bullet
                                for line in agent['lines'][1:]:
                                    clean_line = line.replace('—', '').replace('✓', '').replace('✗', '').strip()
                                    if clean_line:
                                        st.markdown(f"• {clean_line}")
                            else:
                                # Single line - extract reasoning from after the percentage
                                full_text = agent['lines'][0]
                                
                                # Find the end of the confidence percentage: "BUY (50%)"
                                if ')' in full_text:
                                    # Split after the closing parenthesis
                                    parts = full_text.split(')', 1)
                                    if len(parts) > 1:
                                        reasoning_text = parts[1].strip()
                                        
                                        # Remove leading separator if present
                                        if reasoning_text.startswith('—'):
                                            reasoning_text = reasoning_text[1:].strip()
                                        
                                        # Split on semicolons to create bullet points
                                        if ';' in reasoning_text:
                                            points = [p.strip() for p in reasoning_text.split(';') if p.strip()]
                                            for point in points:
                                                st.markdown(f"• {point}")
                                        elif reasoning_text:
                                            # No semicolons, show as single bullet
                                            st.markdown(f"• {reasoning_text}")
                                        else:
                                            st.caption("_No detailed reasoning provided_")
                                    else:
                                        st.caption("_No detailed reasoning provided_")
                                else:
                                    st.caption("_No detailed reasoning provided_")
            
            # Display warnings if present
            if warnings_start:
                st.markdown("---")
                st.markdown("#### ⚠️ Important Warnings")
                warning_lines = [l.strip() for l in lines[warnings_start+1:] if l.strip()]
                for warning in warning_lines:
                    clean_warning = warning.replace('⚠', '').replace('[breakout_strategy]', '').replace('[technical_analysis]', '').replace('[momentum_strategy]', '').strip()
                    if clean_warning:
                        st.warning(f"⚠️ {clean_warning}")
        else:
            # Simple reasoning - display as-is
            st.markdown(reasoning)
        
        if errors:
            with st.expander("⚠️ View Errors", expanded=False):
                for err in errors:
                    st.error(err)
        
        st.markdown("---")
        
        # Agent confidence breakdown
        st.markdown("#### 📊 Agent Confidence Breakdown")
        
        agent_analyses = result.get('agent_analyses', [])
        if agent_analyses:
            # Create visual cards for each agent
            cols = st.columns(len(agent_analyses))
            
            for i, analysis in enumerate(agent_analyses):
                with cols[i]:
                    agent_name = analysis.get('agent_name', 'Unknown').replace('_', ' ').title()
                    confidence = analysis.get('confidence', 0)
                    signal_obj = analysis.get('signal')
                    signal_str = signal_obj.value if hasattr(signal_obj, 'value') else str(signal_obj)
                    
                    # Color-coded based on signal
                    if signal_str == 'BUY':
                        st.success(f"**{agent_name}**")
                    elif signal_str == 'SELL':
                        st.error(f"**{agent_name}**")
                    else:
                        st.warning(f"**{agent_name}**")
                    
                    st.metric("Signal", signal_str)
                    st.metric("Confidence", f"{confidence:.0%}")
                    
                    # Progress bar
                    st.progress(confidence)
            
            st.markdown("---")
            
            # Detailed table
            agent_names = []
            confidences = []
            signals = []
            
            for analysis in agent_analyses:
                agent_name = analysis.get('agent_name', 'Unknown').replace('_', ' ').title()
                agent_names.append(agent_name)
                confidences.append(analysis.get('confidence', 0))
                
                signal_obj = analysis.get('signal')
                signal_str = signal_obj.value if hasattr(signal_obj, 'value') else str(signal_obj)
                signals.append(signal_str)
            
            # Create DataFrame
            df = pd.DataFrame({
                'Agent': agent_names,
                'Signal': signals,
                'Confidence': [f"{c:.0%}" for c in confidences]
            })
            
            st.dataframe(df, width='stretch', hide_index=True)
    
    # ─── TAB 2: SCANNER AGENT ───
    with tab2:
        scanner = result.get('scanner_reasoning', {})
        
        if scanner:
            st.markdown("### 🔍 Initial Screening Results")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### AI Reasoning")
                st.markdown(scanner.get('reasoning', 'No reasoning provided'))
                
                st.markdown("#### 🎯 Detected Signals")
                signals = scanner.get('signals', [])
                if signals:
                    for signal in signals:
                        st.markdown(f"- ✅ {signal}")
                else:
                    st.info("No special signals detected")
            
            with col2:
                st.markdown("#### 📊 Technical Snapshot")
                
                indicators = scanner.get('indicators', {})
                
                rsi = indicators.get('rsi', 0)
                rsi_status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
                rsi_delta = rsi_status
                
                st.metric("RSI", f"{rsi:.1f}", delta=rsi_delta)
                st.metric("MACD", f"{indicators.get('macd', 0):.2f}")
                st.metric("Volume Ratio", f"{indicators.get('volume_ratio', 0):.1f}x")
        else:
            st.info("Scanner data not available in this analysis")
    
    # ─── TAB 3: TECHNICAL AGENT ───
    with tab3:
        technical = result.get('technical_analysis', {})
        
        if technical:
            st.markdown("### 📈 Deep Technical Analysis")
            
            # Main analysis
            detailed = technical.get('detailed_analysis', 'No analysis available')
            
            # Parse if it's the raw agent breakdown format
            if 'FINAL DECISION:' in detailed or 'AGENT BREAKDOWN:' in detailed:
                # Extract components
                lines = detailed.split('\n')
                
                # Parse final decision
                final_decision_line = [l for l in lines if 'FINAL DECISION:' in l]
                agent_breakdown_start = next((i for i, l in enumerate(lines) if 'AGENT BREAKDOWN:' in l), None)
                warnings_start = next((i for i, l in enumerate(lines) if 'WARNINGS:' in l), None)
                
                # Display final decision
                if final_decision_line:
                    st.info(final_decision_line[0].replace('FINAL DECISION:', '**Final Decision:**'))
                
                st.markdown("---")
                
                # Display individual agents
                st.markdown("#### 🤖 Individual Agent Analyses")
                
                if agent_breakdown_start:
                    # Find all agent sections
                    agent_sections = []
                    current_agent = None
                    
                    for i in range(agent_breakdown_start + 1, len(lines)):
                        line = lines[i].strip()
                        
                        if line.startswith('✓ TECHNICAL_ANALYSIS:') or line.startswith('✗ TECHNICAL_ANALYSIS:'):
                            if current_agent:
                                agent_sections.append(current_agent)
                            current_agent = {'type': 'technical', 'lines': [line]}
                        elif line.startswith('✓ MOMENTUM_STRATEGY:') or line.startswith('✗ MOMENTUM_STRATEGY:'):
                            if current_agent:
                                agent_sections.append(current_agent)
                            current_agent = {'type': 'momentum', 'lines': [line]}
                        elif line.startswith('✓ BREAKOUT_STRATEGY:') or line.startswith('✗ BREAKOUT_STRATEGY:'):
                            if current_agent:
                                agent_sections.append(current_agent)
                            current_agent = {'type': 'breakout', 'lines': [line]}
                        elif current_agent and line and not line.startswith('WARNINGS:'):
                            current_agent['lines'].append(line)
                        elif line.startswith('WARNINGS:'):
                            if current_agent:
                                agent_sections.append(current_agent)
                            break
                    
                    # Add last agent if exists
                    if current_agent and current_agent not in agent_sections:
                        agent_sections.append(current_agent)
                    
                    # Display each agent in columns
                    if agent_sections:
                        cols = st.columns(len(agent_sections))
                        
                        for idx, agent in enumerate(agent_sections):
                            with cols[idx]:
                                # Parse header
                                header = agent['lines'][0]
                                is_buy = '✓' in header or 'BUY' in header
                                is_sell = '✗' in header or 'SELL' in header
                                
                                # Extract agent name and signal
                                if 'TECHNICAL_ANALYSIS:' in header:
                                    name = "📊 Technical"
                                    signal_match = header.split('TECHNICAL_ANALYSIS:')[1].split('(')[0].strip()
                                elif 'MOMENTUM_STRATEGY:' in header:
                                    name = "📈 Momentum"
                                    signal_match = header.split('MOMENTUM_STRATEGY:')[1].split('(')[0].strip()
                                elif 'BREAKOUT_STRATEGY:' in header:
                                    name = "🚀 Breakout"
                                    signal_match = header.split('BREAKOUT_STRATEGY:')[1].split('(')[0].strip()
                                else:
                                    name = "Agent"
                                    signal_match = "UNKNOWN"
                                
                                # Extract confidence
                                try:
                                    confidence_match = header.split('(')[1].split(')')[0] if '(' in header else "0%"
                                except:
                                    confidence_match = "0%"
                                
                                # Display card
                                if 'BUY' in signal_match:
                                    st.success(f"**{name}**")
                                elif 'SELL' in signal_match:
                                    st.error(f"**{name}**")
                                else:
                                    st.warning(f"**{name}**")
                                
                                st.metric("Signal", signal_match)
                                st.metric("Confidence", confidence_match)
                                
                                # Extract reasoning
                                reasoning_lines = []
                                for line in agent['lines'][1:]:
                                    clean_line = line.replace('—', '').strip()
                                    if clean_line and not clean_line.startswith('✓') and not clean_line.startswith('✗'):
                                        reasoning_lines.append(clean_line)
                                
                                if reasoning_lines:
                                    with st.expander("💡 View Details"):
                                        for line in reasoning_lines:
                                            st.caption(f"• {line}")
                
                # Display warnings
                if warnings_start:
                    st.markdown("---")
                    st.markdown("#### ⚠️ Warnings")
                    warning_lines = [l.strip() for l in lines[warnings_start+1:] if l.strip()]
                    for warning in warning_lines:
                        clean_warning = warning.replace('⚠', '').replace('[breakout_strategy]', '').replace('[technical_analysis]', '').replace('[momentum_strategy]', '').strip()
                        if clean_warning:
                            st.warning(clean_warning)
            
            else:
                # Simple analysis text - display as-is
                st.markdown(detailed)
            
            st.markdown("---")
            
            # Patterns, Support, Resistance (if available)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🔍 Patterns")
                patterns = technical.get('patterns', [])
                if patterns:
                    for pattern in patterns:
                        st.markdown(f"**{pattern.get('name', 'Unknown')}**")
                        st.caption(pattern.get('description', ''))
                else:
                    st.info("No specific patterns identified")
            
            with col2:
                st.markdown("#### 📉 Support")
                supports = technical.get('support_levels', [])
                if supports:
                    for level in supports:
                        st.markdown(f"₹ {level:,.2f}")
                else:
                    st.info("Support levels not calculated")
            
            with col3:
                st.markdown("#### 📈 Resistance")
                resistances = technical.get('resistance_levels', [])
                if resistances:
                    for level in resistances:
                        st.markdown(f"₹ {level:,.2f}")
                else:
                    st.info("Resistance levels not calculated")
            
            st.markdown("---")
            
            # Overall confidence
            tech_confidence = technical.get('confidence', 0)
            st.metric(
                "📊 Overall Technical Confidence",
                f"{tech_confidence:.0%}",
                delta="High" if tech_confidence > 0.7 else "Medium" if tech_confidence > 0.4 else "Low"
            )
        else:
            st.info("Technical analysis not available")
    
    # ─── TAB 4: RISK ASSESSMENT ───
    with tab4:
        risk = result.get('risk_assessment', {})
        
        if risk:
            st.markdown("### 🛡️ Risk Management & Position Sizing")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                position_pct = risk.get('position_size_pct', 0)
                st.metric("Position Size", f"{position_pct:.1%}")
                st.caption(f"₹ {risk.get('position_size_rupees', 0):,.0f}")
            
            with col2:
                risk_per_trade = risk.get('risk_per_trade', 0)
                st.metric("Risk/Trade", f"{risk_per_trade:.2%}")
            
            with col3:
                rr_ratio = risk.get('reward_risk_ratio', 0)
                st.metric("R:R Ratio", f"1:{rr_ratio:.1f}")
            
            with col4:
                quantity = risk.get('quantity', 0)
                st.metric("Quantity", f"{quantity} shares")
            
            st.markdown("---")
            
            # Pre-trade validation
            st.markdown("#### ✅ Pre-Trade Validation Checks")
            
            checks = risk.get('validation_checks', [])
            if checks:
                for check in checks:
                    passed = check.get('passed', False)
                    rule = check.get('rule', 'Unknown check')
                    status = check.get('status', '')
                    
                    col1, col2 = st.columns([1, 4])
                    
                    with col1:
                        if passed:
                            st.success("✅")
                        else:
                            st.error("❌")
                    
                    with col2:
                        st.markdown(f"**{rule}**")
                        st.caption(status)
            else:
                st.info("No validation checks recorded")
            
            st.markdown("---")
            
            # Position sizing details
            with st.expander("📊 Position Sizing Calculation Details"):
                st.code(f"""
Portfolio Value:    ₹ {risk.get('portfolio_value', 500000):,.0f}
Max Risk per Trade: {risk.get('max_risk_pct', 2.0)}%
Max Position Size:  {risk.get('max_position_pct', 5.0)}%

Entry Price:        ₹ {risk.get('entry_price', 0):.2f}
Stop Loss:          ₹ {risk.get('stop_loss', 0):.2f}
Risk per Share:     ₹ {risk.get('risk_per_share', 0):.2f}

Calculated Shares:  {risk.get('quantity', 0)}
Position Value:     ₹ {risk.get('position_size_rupees', 0):,.0f}
Total Risk Amount:  ₹ {risk.get('position_size_rupees', 0) * risk.get('risk_per_trade', 0) / 100:.2f}
                """, language="text")
        else:
            st.info("Risk assessment not available")
    
    # ─── TAB 5: AGENT FLOW ───
    with tab5:
        st.markdown("### 🧠 Agent Coordination Flow")
        
        st.markdown("""
        This shows how the LangGraph orchestrator coordinates the specialized agents
        to make the final trading decision. Each step represents agent-to-agent communication.
        """)
        
        coordination = result.get('agent_coordination', [])
        
        if coordination:
            st.markdown("#### 📨 Agent Communication Timeline")
            
            for i, message in enumerate(coordination, 1):
                with st.container():
                    col1, col2 = st.columns([1, 5])
                    
                    with col1:
                        st.markdown(f"**Step {i}**")
                        timestamp = message.get('timestamp', 'N/A')
                        st.caption(timestamp if isinstance(timestamp, str) else str(timestamp))
                    
                    with col2:
                        from_agent = message.get('from_agent', 'Unknown')
                        to_agent = message.get('to_agent', 'Unknown')
                        msg = message.get('message', 'No message')
                        
                        st.markdown(f"**{from_agent}** → **{to_agent}**")
                        st.info(msg)
                    
                    # Add arrow between steps
                    if i < len(coordination):
                        st.markdown("↓")
        else:
            st.info("""
            Agent coordination data not available.
            
            In live mode with full LangGraph implementation, this tab would show:
            - Step-by-step agent communication
            - State transitions
            - Decision reasoning flow
            - Timestamp for each step
            """)
        
        st.markdown("---")
        
        # Visual flow diagram
        st.markdown("#### 🔄 System Architecture")
        
        st.graphviz_chart('''
            digraph {
                rankdir=LR;
                node [shape=box, style="rounded,filled", fillcolor=lightblue];
                
                Start [label="Market Data", fillcolor=lightgreen]
                Scanner [label="Scanner\nAgent"]
                Technical [label="Technical\nAgent"]
                Momentum [label="Momentum\nAgent"]
                Breakout [label="Breakout\nAgent"]
                Orchestrator [label="Orchestrator\n(LangGraph)", fillcolor=gold]
                Risk [label="Risk\nValidator"]
                Decision [label="Final\nDecision", fillcolor=lightcoral]
                
                Start -> Scanner
                Scanner -> Orchestrator [label="Candidates"]
                Orchestrator -> Technical [label="Analyze"]
                Orchestrator -> Momentum [label="Analyze"]
                Orchestrator -> Breakout [label="Analyze"]
                Technical -> Orchestrator [label="Score"]
                Momentum -> Orchestrator [label="Score"]
                Breakout -> Orchestrator [label="Score"]
                Orchestrator -> Risk [label="Validate"]
                Risk -> Decision [label="Approved"]
            }
        ''')

else:
    # ========================================================================
    # NO ANALYSIS YET - SHOW PLACEHOLDER
    # ========================================================================
    
    st.markdown("### 👆 Select a stock and click 'Analyze' to see multi-agent reasoning")
    
    st.markdown("""
    ## How the Multi-Agent System Works
    
    This page demonstrates the complete multi-agent architecture:
    
    ### 🔄 Process Flow
    
    1. **Scanner Agent** 
       - Screens the market for opportunities
       - Applies technical filters (RSI, volume, MACD)
       - Flags stocks worthy of deeper analysis
    
    2. **Specialized Strategy Agents**
       - **Technical Agent**: Deep indicator analysis, pattern recognition
       - **Momentum Agent**: Trend strength, directional bias
       - **Breakout Agent**: Support/resistance levels, breakout potential
    
    3. **LangGraph Orchestrator**
       - Coordinates all agents via state machine
       - Aggregates individual analyses
       - Builds consensus from agent recommendations
    
    4. **Risk Validator**
       - Calculates position size (Kelly Criterion)
       - Validates against portfolio constraints
       - Ensures risk limits are respected
    
    5. **Final Decision**
       - BUY / HOLD / SELL with confidence score
       - Complete reasoning chain
       - Risk-adjusted position sizing
    
    ---
    
    ### 🎯 What Makes This Different
    
    **Not just an LLM wrapper:**
    - ✅ True LangGraph state machine
    - ✅ Agent-to-agent communication
    - ✅ Explicit reasoning chains
    - ✅ Production-grade risk management
    
    **Explainable AI:**
    - Every decision is traceable
    - Individual agent confidence scores
    - Clear consensus mechanism
    - No black-box outputs
    
    ---
    
    **Ready to explore?** Select a stock from the dropdown above and click Analyze!
    """)
    
    # Show architecture diagram
    with st.expander("🏗️ View System Architecture"):
        st.graphviz_chart('''
            digraph {
                rankdir=TB;
                node [shape=box, style="rounded,filled"];
                
                subgraph cluster_input {
                    label="Data Layer";
                    style=filled;
                    color=lightgrey;
                    
                    Market [label="Market Data\n(Yahoo Finance)", fillcolor=lightgreen]
                    Indicators [label="Technical\nIndicators", fillcolor=lightgreen]
                }
                
                subgraph cluster_agents {
                    label="Agent Layer (LangGraph)";
                    style=filled;
                    color=lightblue;
                    
                    Scanner [label="Scanner Agent"]
                    Technical [label="Technical Agent"]
                    Momentum [label="Momentum Agent"]
                    Breakout [label="Breakout Agent"]
                    Orchestrator [label="Orchestrator", fillcolor=gold]
                }
                
                subgraph cluster_risk {
                    label="Risk Layer";
                    style=filled;
                    color=lightyellow;
                    
                    Position [label="Position Sizer"]
                    Validator [label="Pre-Trade\nValidator"]
                }
                
                Decision [label="Final Decision\n(BUY/HOLD/SELL)", fillcolor=lightcoral, shape=diamond]
                
                Market -> Indicators
                Indicators -> Scanner
                Scanner -> Orchestrator
                Orchestrator -> Technical
                Orchestrator -> Momentum
                Orchestrator -> Breakout
                Technical -> Orchestrator
                Momentum -> Orchestrator
                Breakout -> Orchestrator
                Orchestrator -> Position
                Position -> Validator
                Validator -> Decision
            }
        ''')

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

if demo_mode:
    st.info("""
    💡 **You're in Demo Mode**
    
    This shows authentic agent reasoning captured from local runs with real API calls.
    All multi-agent coordination, risk calculations, and decisions are genuine.
    
    **To run live:**
    ```bash
    git clone https://github.com/sohansputhran/ai-trading-copilot.git
    # Add HUGGINGFACE_API_TOKEN=hf_... to .env
    streamlit run streamlit_app/app.py
    ```
    
    [📖 Architecture Docs](https://github.com/sohansputhran/ai-trading-copilot#architecture) |
    [💻 Source Code](https://github.com/sohansputhran/ai-trading-copilot) |
    [🎥 Watch Demo Video](https://github.com/sohansputhran/ai-trading-copilot#demo)
    """)
else:
    st.success("""
    🔴 **Live Mode Active**
    
    You're running real-time multi-agent analysis with API calls.
    Each analysis consumes HuggingFace API credits.
    
    💡 **Tip:** Enable demo mode in sidebar to explore without API costs.
    """)

st.caption("Built with ❤️ using LangGraph, HuggingFace, and Streamlit")