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

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.agents.scanner_agent import MarketScanner, NIFTY_50_SAMPLE
from src.data_pipeline.collector import MarketDataCollector
from src.data_pipeline.indicators import SimpleTechnicalIndicators

# Page config
st.set_page_config(
    page_title="AI Trading Scanner",
    layout="wide"
)

# Title
st.title("AI Trading Scanner")
st.markdown("*Powered by free HuggingFace AI (Mistral-7B)*")

# Sidebar - Input
st.sidebar.header("Scanner Settings")

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
        
        # Run scan with progress updates
        results = []
        for i, symbol in enumerate(symbols):
            status_text.text(f"Scanning {symbol}... ({i+1}/{len(symbols)})")
            result = scanner.scan_stock(symbol)
            if result:
                results.append(result)
            progress_bar.progress((i + 1) / len(symbols))
        
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        st.success(f"Scan complete! Found {len(results)} interesting stocks")
        
        if results:
            st.subheader("Interesting Stocks Found")
            
            # Show each result
            for i, result in enumerate(results, 1):
                with st.expander(f"**{i}. {result['symbol']}** - {result['price']:.2f}", expanded=True):
                    
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
                        st.info(result['analysis'])
                    
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
                            yaxis_title="Price (₹)",
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.warning(f"Could not load chart: {str(e)}")
        
        else:
            st.info("No stocks matched the criteria. Try scanning more stocks or check back later!")

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
    - **HuggingFace Mistral-7B** (Free AI model)
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
