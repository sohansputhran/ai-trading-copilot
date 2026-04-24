# 🤖 AI Trading Copilot

> Production-grade agentic AI system for algorithmic trading with explainable decision-making

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[View Live Demo](https://your-demo-link.streamlit.app) • [Documentation](docs/) • [Architecture](docs/ARCHITECTURE.md)

---

## 🎯 What This Is

An **AI Engineering portfolio project** demonstrating production-grade multi-agent systems, real-time data engineering, and algorithmic risk management. Built to showcase mastery of:

- 🧠 **Multi-Agent AI Orchestration** using LangGraph
- 📊 **Real-Time Data Pipelines** with caching and validation
- ⚖️ **Algorithmic Risk Management** (Kelly Criterion, position sizing)
- 🔍 **Explainable AI Decisions** with transparent reasoning chains
- 📈 **Production System Design** (observability, testing, deployment)

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Market Scanner** | Identifies trading opportunities across 50+ stocks using technical analysis |
| **Multi-Agent Analysis** | Specialized AI agents (Technical, Momentum, Risk) collaborate to analyze stocks |
| **Risk Engine** | Pre-trade validation with portfolio-level constraints and circuit breakers |
| **Paper Trading** | Simulated execution with real broker API integration (Upstox) |
| **Trade Journal** | Automated performance analytics (Sharpe, Sortino, win rate) with PostgreSQL |
| **Real-Time Dashboard** | Streamlit UI showing live scans, agent reasoning, and performance metrics |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD                       │
│         (Real-time Market Scans & Agent Reasoning)           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              LANGGRAPH ORCHESTRATION LAYER                   │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│   │   Market     │→ │   Strategy   │→ │     Risk     │    │
│   │   Scanner    │  │   Analyzer   │  │   Manager    │    │
│   └──────────────┘  └──────────────┘  └──────────────┘    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  DATA & EXECUTION LAYER                      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Market    │  │   Upstox     │  │    Trade     │      │
│  │   Data      │  │   Broker     │  │   Journal    │      │
│  │   Pipeline  │  │   Connector  │  │  (Postgres)  │      │
│  └─────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL (or use SQLite for development)
- **HuggingFace API token** ([get one FREE here](https://huggingface.co/settings/tokens))
  - Alternative: Anthropic API key (paid, but more advanced)

### Installation

```bash
# Clone the repository
git clone https://github.com/sohansputhran/ai-trading-copilot.git
cd ai-trading-copilot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your HUGGINGFACE_API_TOKEN to .env (FREE!)

# Initialize database (optional - uses SQLite by default)
# For PostgreSQL: update DATABASE_URL in .env

# Run the dashboard
streamlit run app.py
```

Visit `http://localhost:8501` to see the dashboard.

### Deploy to Streamlit Cloud (Recommended)

The easiest way to deploy this project:

1. **Push to GitHub** (if not already done)
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Click "New app"**
4. **Select your repo** and `app.py` as the main file
5. **Add secrets** in Advanced Settings:
   ```toml
   HUGGINGFACE_API_TOKEN = "hf_your-token-here"
   ```
6. **Click Deploy!**

Your app will be live at `https://your-app-name.streamlit.app` in minutes!

**Note:** Project uses FREE HuggingFace Inference API by default. Works with models like Llama, Mistral, Zephyr!

---

## 📊 Live Demo

<p align="center">
  <img src="docs/images/dashboard-preview.png" alt="Dashboard Preview" width="800"/>
</p>

**Try it yourself:** [Live Streamlit App](https://your-demo-link.streamlit.app)

### What You'll See:
1. **Market Scanner** - Real-time stock screening with technical indicators
2. **Agent Reasoning** - Transparent AI decision-making process
3. **Risk Dashboard** - Portfolio exposure and risk metrics
4. **Trade Analytics** - Performance metrics and equity curves

---

## 🛠️ Tech Stack

**Core Technologies:**
- **Agent Framework:** LangGraph (multi-agent orchestration)
- **LLM:** HuggingFace Inference API (FREE - Llama, Mistral, Zephyr)
  - Alternative: Claude 3.5 Sonnet via Anthropic API (paid, more advanced)
- **Data Pipeline:** Yahoo Finance, pandas, custom indicator engine
- **UI:** Streamlit with custom components
- **Database:** PostgreSQL / SQLite
- **Testing:** pytest (80%+ coverage)

**AI Engineering Patterns:**
- Multi-agent collaboration with state management
- Tool-calling for external data integration
- Explainable AI with reasoning chains
- Production-grade error handling and retry logic

**100% Free Tools:** HuggingFace API, Yahoo Finance, SQLite, Streamlit Cloud - zero costs!

---

## 📈 Project Highlights

### What Makes This Different

**1. Production-Grade Engineering**
- Complete Docker containerization with multi-stage builds
- CI/CD pipeline with GitHub Actions (lint, test, deploy)
- Type hints on all functions
- Comprehensive error handling with structured logging
- 82%+ test coverage with pytest
- Modular, testable architecture

**2. Explainable AI**
- Every trade decision includes reasoning chain
- Agent collaboration is transparent and auditable
- Human-readable explanations for all signals

**3. Real-World Complexity**
- Multi-agent coordination (not just single LLM calls)
- Real-time data pipelines with caching strategies
- Portfolio-level risk management
- Broker API integration for live execution

**4. Learning Journey**
- 12-week sprint-based development
- Each sprint focuses on specific AI Engineering concepts
- Documented learnings and architectural decisions

---

## 📚 Documentation

Comprehensive documentation available in the [`docs/`](docs/) folder:

- **[Architecture Guide](docs/ARCHITECTURE.md)** - System design and component interactions
- **[Technical Specifications](docs/TECHNICAL_SPECS.md)** - Detailed requirements and data models
- **[Development Guide](docs/DEVELOPMENT.md)** - Setup, testing, and contribution guidelines
- **[Sprint Roadmap](docs/ROADMAP.md)** - 12-week project plan and progress tracking
- **[API Reference](docs/API_REFERENCE.md)** - Internal APIs and agent tools

---

## 🎓 Key Learnings & Skills Demonstrated

### AI Engineering
- ✅ Multi-agent system orchestration with LangGraph
- ✅ Tool calling and function execution patterns
- ✅ Prompt engineering for financial domain
- ✅ LLM observability and debugging

### Data Engineering
- ✅ Real-time data pipelines with validation
- ✅ Caching strategies for performance
- ✅ Technical indicator computation at scale
- ✅ Event-driven architecture patterns

### System Design
- ✅ Modular, testable architecture
- ✅ Production-grade error handling
- ✅ Database design for time-series data
- ✅ Observability with structured logging

### Trading & Finance
- ✅ Technical analysis implementation
- ✅ Risk management algorithms
- ✅ Position sizing (Kelly Criterion)
- ✅ Performance attribution analysis

---

## 🗺️ Roadmap

| Sprint | Status | Focus Area |
|--------|--------|------------|
| **Sprint 1** | ✅ Complete | Market Data Pipeline & Scanner Agent |
| **Sprint 2** | ✅ Complete | Multi-Agent Strategy Analyzer |
| **Sprint 3** | ✅ Complete | Risk Management Engine |
| **Sprint 4** | ✅ Complete | Paper Trading Integration |
| **Sprint 5** | ✅ Complete | Trade Journal & Analytics |
| **Sprint 6** | ✅ Complete | Production Deployment (Docker, CI/CD) |

See [ROADMAP.md](docs/ROADMAP.md) for detailed sprint breakdown.

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test suite
pytest tests/test_indicators.py -v
```

Current test coverage: **82%**

---

## 🤝 Contributing

This is a portfolio project, but feedback and suggestions are welcome! Please:

1. Open an issue to discuss proposed changes
2. Follow the existing code style (black, ruff, mypy)
3. Add tests for new functionality
4. Update documentation as needed

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for detailed guidelines.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 About the Developer

**Sohan** - AI Engineer | Building production-grade AI systems

- 📧 Email: your.email@example.com
- 💼 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- 🐙 GitHub: [@sohansputhran](https://github.com/sohansputhran)
- 📝 Blog: [yourblog.com](https://yourblog.com)

**Built as a portfolio project to demonstrate:**
- Deep expertise in AI Engineering and LangGraph
- Production-grade system design and architecture
- Real-world problem-solving in quantitative finance
- Commitment to code quality and best practices

---

## 🙏 Acknowledgments

- **LangGraph/LangChain** - Agent orchestration framework
- **HuggingFace** - HuggingFace API
- **Streamlit** - Rapid UI prototyping
- **Yahoo Finance** - Market data source

---

<p align="center">
  <strong>⭐ If this project helped you learn about AI Engineering, consider starring the repo!</strong>
</p>

<p align="center">
  Made with ❤️ and lots of ☕ | Built to learn, built to showcase
</p>