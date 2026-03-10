# AI Trading Copilot - Sprint 1

Building an intelligent trading system using AI agents.

## Sprint 1 Goal
Build market data pipeline and basic scanner agent.

## What We're Building This Sprint
1. ✅ Market data collector (fetch stock prices)
2. ⏳ Technical indicators (RSI, MACD, etc.)
3. ⏳ Scanner agent (find trading opportunities)
4. ⏳ Streamlit dashboard (show results)

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create .env File
```bash
cp .env.example .env
```

Edit `.env` and add your Anthropic API key:
```
ANTHROPIC_API_KEY=sk-ant-...
```

Get your key from: https://console.anthropic.com/

### 3. Test Market Data Collector
```bash
python -m src.data_pipeline.collector
```

You should see Reliance stock data fetched successfully.

## Project Structure (Sprint 1)
```
ai-trading-copilot/
├── src/
│   ├── utils/
│   │   └── config.py          # Loads API key from .env
│   └── data_pipeline/
│       └── collector.py       # Fetches stock data
├── requirements.txt           # Python packages
├── .env.example              # Template for API keys
└── README.md                 # This file
```

## Next Steps
1. Add technical indicators (RSI, MACD)
2. Build scanner agent with LangGraph
3. Create Streamlit dashboard

---

**Current Status**: Setting up foundation ✅
