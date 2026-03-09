# AI Trading Copilot

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-green.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Production-grade AI trading copilot using LangGraph multi-agent orchestration. Features real-time market scanning, explainable AI decision-making, and algorithmic risk management.

## 🎯 Project Overview

An intelligent trading system that combines:
- **Multi-Agent AI** - LangGraph orchestration with specialized agents
- **Real-time Analysis** - Market scanning with technical indicators
- **Risk Management** - Algorithmic position sizing and validation
- **Explainable AI** - Human-readable reasoning for every decision

Built as a portfolio project to showcase AI Engineering expertise.

## 🏗️ Architecture
```
Streamlit Dashboard
    ↓
LangGraph Multi-Agent System
    - Market Scanner Agent
    - Strategy Analyzer Agent  
    - Risk Manager Agent
    ↓
Data & Execution Layer
    - Market Data Pipeline
    - Upstox Broker API
    - Trade Journal (PostgreSQL)
```

## ✨ Features

- 🔍 **Intelligent Market Scanning** - Finds high-potential stocks using technical analysis
- 🤖 **Multi-Agent Decision System** - Coordinated AI agents with specialized strategies
- 🛡️ **Risk Management** - Kelly Criterion position sizing and pre-trade validation
- 📊 **Real-time Dashboard** - Professional Streamlit interface with live updates
- 📝 **Automated Journaling** - Every trade logged with reasoning and metrics
- 🎯 **Explainable AI** - Clear explanations for every trading decision

## 🛠️ Tech Stack

**Core:**
- Python 3.11+
- LangGraph (Multi-agent orchestration)
- LangChain (Agent tools)
- Claude (Anthropic LLM)

**Data & Analysis:**
- pandas, numpy (Data processing)
- TA-Lib (Technical indicators)
- yfinance (Market data)

**Backend:**
- FastAPI (REST API)
- PostgreSQL (Trade journal)
- Redis (Caching - optional)

**Frontend:**
- Streamlit (Dashboard)
- Plotly (Charts)

**DevOps:**
- Docker & docker-compose
- GitHub Actions (CI/CD)
- pytest (Testing)

## 🚀 Quick Start

### Prerequisites
```bash
- Python 3.11+
- pip or Poetry
- Git
```

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/ai-trading-copilot.git
cd ai-trading-copilot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Run the Dashboard
```bash
streamlit run streamlit_app/app.py
```

Open http://localhost:8501 in your browser.

## 📚 Project Structure
```
ai-trading-copilot/
├── src/
│   ├── agents/              # LangGraph agents
│   ├── data_pipeline/       # Market data collection
│   ├── risk_management/     # Position sizing & validation
│   ├── execution/           # Order management
│   ├── journal/             # Trade logging & analytics
│   └── utils/               # Shared utilities
├── streamlit_app/           # Dashboard UI
├── configs/                 # Configuration files
├── tests/                   # Unit & integration tests
├── docs/                    # Documentation
└── notebooks/               # Learning notebooks
```

## 🎓 Learning Journey

This project was built as a 12-week learning journey through:
1. Real-time data engineering
2. LangGraph multi-agent orchestration
3. Algorithmic risk management
4. Production system design

See [PROJECT_MASTER.md](docs/PROJECT_MASTER.md) for the complete roadmap.

## 📊 Current Status

**Sprint 1: Market Data Pipeline & Scanner Agent** ✅ (In Progress)
- [x] Project setup and architecture
- [ ] Market data collector
- [ ] Technical indicator engine
- [ ] Scanner agent
- [ ] Streamlit dashboard

## 🤝 Contributing

This is a portfolio/learning project, but suggestions and feedback are welcome!

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Market data from [Yahoo Finance](https://finance.yahoo.com/)
- Inspired by quantitative trading and AI agent research

## 📧 Contact

**Sohan** - AI Engineering Portfolio Project

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Portfolio: [Your Website](https://yourwebsite.com)

---

⭐ Star this repo if you find it interesting!