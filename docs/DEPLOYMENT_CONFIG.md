# Streamlit Cloud Deployment Configuration

## Setting Demo Mode

To enable demo mode for Streamlit Cloud (where SQLite is not supported), set the following environment variable in your Streamlit Cloud app settings:

```
STREAMLIT_CLOUD=true
```

### How to set environment variables in Streamlit Cloud:

1. Go to your app settings in Streamlit Cloud
2. Navigate to "Secrets" section
3. Add the following:

```toml
STREAMLIT_CLOUD = "true"
```

Alternatively, you can set it in your `secrets.toml` file:

```toml
STREAMLIT_CLOUD = "true"
```

## What Demo Mode Does

When `STREAMLIT_CLOUD=true`:

1. **Portfolio Page** (`pages/2_portfolio.py`):
   - Displays realistic sample portfolio data with 4 open positions
   - Shows dummy P&L calculations and metrics
   - Disables live price fetching and position closing
   - Adds banner: "📊 Demo Mode — You're viewing sample portfolio data"

2. **Analytics Page** (`pages/3_analytics.py`):
   - Generates 20 sample closed trades with realistic distribution
   - Shows computed performance metrics (win rate, profit factor, Sharpe, etc.)
   - Displays equity curve from dummy data
   - Shows per-strategy breakdown
   - Adds banner: "📊 Demo Mode — You're viewing sample analytics data"

## Sample Data Characteristics

### Portfolio (Open Positions)
- 4 positions: RELIANCE, TCS, HDFCBANK, INFY
- Mix of winning and losing positions
- Realistic entry/exit prices for Indian stocks
- Total unrealized P&L: ~₹5,000
- Demonstrates all UI features and visualizations

### Analytics (Closed Trades)
- 20 historical trades
- 70% win rate (14 winners, 6 losers)
- Multiple strategies: momentum_breakout, technical_analysis, breakout_strategy
- Profit Factor: ~2.3
- Avg R-Multiple: ~1.2
- Total P&L: ~₹48,000

## Local Development

When running locally with database support, the app automatically detects the presence of `data/trades.db` and uses real data instead of dummy data.

No environment variable needed for local development — just run:

```bash
streamlit run app.py
```

## Files Modified

1. `pages/2_portfolio.py` - Added demo mode with dummy portfolio data
2. `pages/3_analytics.py` - Added demo mode with dummy analytics data

Both files automatically detect deployment mode and switch between real/dummy data accordingly.
