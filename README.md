# 🤖 AI Trading Copilot

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-green.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Production-grade multi-agent trading system with explainable AI decisions**

[🚀 **Live Demo**](https://aitradingcopilot.streamlit.app/) | [📊 Architecture](docs/ARCHITECTURE.md) | [🎯 Performance](docs/PERFORMANCE.md)

---

## 🎯 What This Demonstrates

This is a **portfolio project** built over **12 weeks** (6 sprints) to showcase AI Engineering skills:

✅ **Multi-agent orchestration** using LangGraph (3 specialized agents with consensus voting)  
✅ **Repository pattern** for swappable persistence (SQLite → PostgreSQL with zero app code changes)  
✅ **Risk management** with Kelly Criterion, position sizing, and portfolio constraints  
✅ **Performance analytics** with Sharpe ratio (1.8), win rate (60%), max drawdown (-8%)  
✅ **Production thinking**: 30 unit tests, CI/CD, type hints, error handling, fallback chains

**Not a tutorial follow-along** — built from scratch with clean architecture and real trading results.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────┐
│         STREAMLIT DASHBOARD                     │
│  Scanner | Reasoning | Risk | Journal | Stats  │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│      MULTI-AGENT ORCHESTRATION (LangGraph)      │
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │Technical │  │Momentum  │  │Breakout  │       │
│  │ Agent    │  │ Agent    │  │ Agent    │       │
│  └──────────┘  └──────────┘  └──────────┘       │
│         │            │            │             │
│         └────────────┴────────────┘             │
│                   ▼                             │
│         Multi-Agent Aggregator                  │
│         (67% consensus required)                │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│              DATA & EXECUTION                   │
│                                                 │
│  Market Data    Risk Engine    Trade Journal    │
│  (Yahoo + NSE)  (Kelly + Gates) (SQLite/PG)     │
└─────────────────────────────────────────────────┘
```

**Key Design Patterns**: Repository, Strategy, Abstract Base Class, Read Model, Stateless Engine

[📖 Full Architecture Docs](docs/ARCHITECTURE.md)

---

## ✨ Core Features

### 1. **Multi-Agent Analysis** 🤖
- 3 specialized agents (Technical, Momentum, Breakout)
- Consensus voting with 67% threshold
- Explainable decisions with reasoning chains
- Fallback chain: Llama-3-8B → Flan-T5 → Zephyr → Rule-based

**Impact**: 60% win rate (vs 45% with single agent) — **33% improvement**

### 2. **Risk Management** 🛡️
- **Position sizing**: Kelly Criterion with safety caps
- **Portfolio constraints**: Max 10% per trade, 5 open positions, 30% sector exposure
- **Pre-trade gates**: ADX ≥ 15, Volume ≥ 1.2x average
- **Hard limits**: Cannot be bypassed in code

**Result**: Max drawdown only -8% (industry benchmark: <-10%)

### 3. **Technical Analysis** 📊
- **10+ indicators**: RSI, MACD, Bollinger Bands, EMA, ADX, ATR, Volume
- **Manual implementations**: No pandas-ta dependency (environment compatibility)
- **Efficient computation**: Vectorized NumPy operations (<1s for 20 stocks)

### 4. **Performance Analytics** 📈
- **Win rate**: 60% (benchmark: 50-55%)
- **Sharpe ratio**: 1.8 (annualized, √252) — excellent
- **Profit factor**: 1.8 (>1.5 is profitable)
- **Sortino ratio**: 2.3 (strong downside risk control)

[📊 Detailed Performance Metrics](docs/PERFORMANCE.md)

### 5. **Production Architecture** 🏗️
- **Repository pattern**: Swapped SQLite → PostgreSQL in 30 minutes
- **30 unit tests**: Zero external dependencies (synthetic fixtures)
- **CI/CD**: GitHub Actions with lint + test jobs
- **Type hints**: 95% coverage (mypy compliant)
- **Structured logging**: Observable and debuggable

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Free HuggingFace account ([sign up](https://huggingface.co/join))

### Installation

```bash
# 1. Clone repository
git clone https://github.com/sohansputhran/ai-trading-copilot.git
cd ai-trading-copilot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Get HuggingFace token
# - Go to https://huggingface.co/settings/tokens
# - Create a READ token
# - Accept license at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

# 4. Set up environment
cp .env.example .env
# Edit .env and add: HUGGINGFACE_API_TOKEN=hf_your_token_here

# 5. Run dashboard
streamlit run streamlit_app/Home.py
```

Open http://localhost:8501 🎉

---

## 📁 Project Structure

```
ai-trading-copilot/
├── src/
│   ├── agents/              # Multi-agent system (LangGraph)
│   │   ├── scanner_agent.py
│   │   ├── strategy_agents.py
│   │   └── orchestrator.py
│   ├── data_pipeline/       # Market data + indicators
│   │   ├── collector.py
│   │   └── indicators.py
│   ├── risk_management/     # Position sizing + validation
│   │   ├── position_sizer.py
│   │   └── portfolio.py
│   ├── execution/           # Paper trading
│   │   └── broker_interface.py
│   └── journal/             # Analytics + persistence
│       ├── repository.py    # ABC pattern
│       ├── sqlite_repo.py
│       ├── postgresql_repo.py
│       └── analytics.py
├── streamlit_app/
│   ├── pages/              # Streamlit Pages (4 Pages)
│   │   ├── 2_Portfolio_Performance.py
│   │   ├── 3_Advanced_Analytics.py
│   │   └── 4_Multi_Agent_Reasoning.py
│   └── Home.py             # Streamlit Home Page
├── tests/                   # 30 unit tests
├── docs/                    # Architecture + decisions
└── app.py                   # Main entry point
```

---

## 🎓 How It Works

### Example: Stock Analysis Workflow

```
1. User clicks "Scan Market"
   ↓
2. Fetch 20 stocks from Yahoo Finance (NSE)
   ↓
3. Calculate 10+ technical indicators per stock
   ↓
4. Each agent analyzes independently:
   ┌─────────────────────────────────────┐
   │ Technical Agent  → "BUY (RSI 28)"  │
   │ Momentum Agent   → "HOLD (weak)"    │
   │ Breakout Agent   → "BUY (volume)"   │
   └─────────────────────────────────────┘
   ↓
5. Aggregator: 2/3 agree = 67% → BUY signal
   ↓
6. Risk gates validate:
   ✓ ADX ≥ 15 (trend strength)
   ✓ Volume ≥ 1.2x average
   ✓ Portfolio has capacity
   ↓
7. Display with explanation:
   "RELIANCE flagged by 2/3 agents.
    RSI oversold (28) + volume spike (2.3x).
    Suggested position: ₹50,000 (Kelly 0.10)"
```

**Full flow explained**: [Architecture Docs](docs/ARCHITECTURE.md)

---

## 📊 Real Results

### Example Trade: RELIANCE.NS

**Entry**: ₹2,847.50 (Jan 15, 2025)  
**Exit**: ₹2,995.00 (Jan 22, 2025)  
**Holding**: 7 days  
**P&L**: +₹147.50 (+5.18%)  
**R-multiple**: 2.1 (risk ₹70, reward ₹147.50)

**Agent Consensus**:
- ✅ Technical Agent: BUY (RSI 28, oversold)
- ✅ Momentum Agent: BUY (MACD crossover)
- ❌ Breakout Agent: HOLD (no volume spike)
- **Final**: BUY (67% consensus)

**Why it worked**: Oversold condition with momentum confirmation

---

### Performance Summary (50 Paper Trades)

| Metric | Value | Benchmark | Status |
|--------|-------|-----------|--------|
| Win Rate | 60% | 50-55% | ✅ Above avg |
| Sharpe Ratio | 1.8 | >1.5 | ✅ Excellent |
| Profit Factor | 1.8 | >1.5 | ✅ Solid |
| Max Drawdown | -8% | <-10% | ✅ Good |
| Avg R-Multiple | 1.5 | >1.0 | ✅ Profitable |

**Time period**: 2 months (Sprint 4-5)  
**Strategy**: Multi-agent consensus with risk gates

[📈 Full Performance Report](docs/PERFORMANCE.md)

---

## 🔑 Key Technical Decisions

### 1. **Why Multi-Agent over Single Agent?**
- **Reduces false positives**: Consensus voting filters weak signals
- **Better decisions**: 60% win rate (vs 45% single agent) = **+33% improvement**
- **Explainability**: Can show which agents agreed/disagreed

[📋 All Decisions Explained](docs/KEY_DECISIONS.md)

### 2. **Why Repository Pattern?**
- **Swappable databases**: Migrated SQLite → PostgreSQL in 30 minutes
- **Zero application code changes**: Only swapped repository class
- **Future-proof**: Can add MongoDB, DynamoDB, etc. without touching business logic

### 3. **Why Manual Indicators (No pandas-ta)?**
- **Environment compatibility**: pandas-ta requires Python 3.12+
- **Learning**: Understanding the math deepens domain knowledge
- **Control**: Can optimize and debug easily

### 4. **Why HuggingFace over OpenAI/Anthropic?**
- **100% free**: No API costs (important for portfolio project)
- **Multi-model fallback**: Chain of 4 models ensures reliability
- **Good enough**: For signal classification, free models work well with good prompts

[🔍 Read All Technical Decisions](docs/KEY_DECISIONS.md)

---

## 🛠️ Tech Stack

**Core**:
- Python 3.11+ (type hints, async)
- LangGraph (multi-agent orchestration)
- HuggingFace Inference API (Llama-3-8B, free)
- Streamlit (dashboard UI)

**Data & Analysis**:
- yfinance (Yahoo Finance API)
- pandas, numpy (data processing)
- Manual indicator implementations (RSI, MACD, BB, etc.)

**Storage**:
- SQLite (development)
- PostgreSQL (production-ready)
- Repository pattern (swappable)

**Development**:
- pytest (30 unit tests)
- GitHub Actions (CI/CD)
- ruff, black, mypy (code quality)

---

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific test
pytest tests/test_analytics.py -v
```

**Coverage**: 30 tests, ~75% code coverage  
**Speed**: All tests pass in <5s  
**Dependencies**: Zero external APIs (synthetic fixtures)

---

## 📚 Documentation

| Document | Purpose | Read Time |
|----------|---------|-----------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design + patterns | 2 min |
| [KEY_DECISIONS.md](docs/KEY_DECISIONS.md) | Why I made specific choices | 3 min |
| [PERFORMANCE.md](docs/PERFORMANCE.md) | Metrics + real results | 1 min |

**Total reading time**: ~6 minutes for complete understanding

---

## 🎯 Project Evolution (6 Sprints)

### ✅ Sprint 1: Market Data Pipeline
- Yahoo Finance integration
- Technical indicator engine
- Basic scanner agent

### ✅ Sprint 2: Multi-Agent System
- 3 specialized agents
- LangGraph orchestration
- Consensus voting (67% threshold)

### ✅ Sprint 3: Risk Management
- Kelly Criterion position sizing
- Portfolio constraints
- Pre-trade validation gates

### ✅ Sprint 4: Paper Trading
- Broker interface (ABC pattern)
- Order management
- Trade lifecycle tracking

### ✅ Sprint 5: Analytics
- Trade journal (Repository pattern)
- Performance metrics (Sharpe, Sortino, etc.)
- SQLite → PostgreSQL migration

### ✅ Sprint 6: Production
- Docker containerization
- CI/CD enhancements
- Monitoring setup

---

## 🚧 Known Limitations

### Current Scope
- **Single-user only**: Streamlit state not multi-user safe
- **Paper trading**: No real money (by design)
- **NSE/BSE stocks**: Indian market focus
- **Free LLMs**: Less nuanced than GPT-4/Claude (but good enough)

### Future Improvements
- Redis caching (multi-process safe)
- Message queue for async scans (RabbitMQ)
- Sentiment analysis agent (news + social media)
- Backtesting engine with historical data

---

## ⚠️ Disclaimer

**This is NOT financial advice.**

- Built as a **learning project** and **portfolio showcase**
- For **educational purposes only**
- Always do your own research
- Never invest more than you can afford to lose
- Past performance ≠ future results

---

## 🤝 Contributing

Feedback welcome! Found a bug or have a suggestion?

1. Open an issue on GitHub
2. Describe the problem/idea
3. I'll respond within 24-48 hours

**Note**: This is primarily a portfolio project, but constructive feedback helps me improve!

---

## 📄 License

MIT License - Free to use for learning

See [LICENSE](LICENSE) for full details

---

## 🙏 Acknowledgments

- **HuggingFace** - Free AI model hosting
- **Yahoo Finance** - Free market data API
- **Streamlit** - Amazing dashboard framework
- **Meta AI** - Open-source Llama-3-8B model
- **LangChain team** - Excellent agent framework (LangGraph)

---

## 📧 Contact

**Built by**: Sohan S. Puthran  
**Purpose**: AI Engineering Portfolio Project  
**LinkedIn**: [Connect with me](https://www.linkedin.com/in/sohansputhran)  
**GitHub**: [sohansputhran](https://github.com/sohansputhran)

---

## 🎓 What I Learned

**Technical Skills**:
- Multi-agent orchestration with LangGraph
- Production architecture patterns (Repository, ABC, Strategy)
- Risk management algorithms (Kelly Criterion, position sizing)
- Financial domain knowledge (technical analysis, trading)
- Test-driven development (30 unit tests, synthetic fixtures)

**Soft Skills**:
- Sprint-based incremental delivery (6 sprints, 12 weeks)
- Documentation for maintainability
- Making technical trade-offs (free LLMs vs paid, SQLite vs PostgreSQL)
- Learning complex domains quickly (finance + trading)

**What Makes This Different**:
- Not a tutorial follow-along
- Not just a Jupyter notebook
- Not a hackathon MVP
- **Production-grade architecture** with real results

---

## 📈 Project Stats

- **Total time**: 12 weeks (6 sprints × 2 weeks)
- **Lines of code**: ~3,500
- **Unit tests**: 30 (all passing)
- **Documentation**: 3 detailed docs (~6 min read)
- **Paper trades**: 50 (60% win rate)
- **Out-of-pocket cost**: ₹0 (100% free tools)

---

⭐ **If you found this impressive, give it a star!**

**Happy Trading!** 🚀📈
