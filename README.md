# 🤖 AI Trading Copilot - Sprint 1

An intelligent trading system that uses **free AI** to scan stocks and identify trading opportunities.

> **100% Free** - No API costs, uses open-source HuggingFace models

---

## 🎯 What This Does

Scans stocks and uses AI to find potential trading opportunities based on technical indicators:

- 📊 **Calculates indicators** - RSI, MACD, Bollinger Bands, Volume
- 🤖 **AI analysis** - Mistral-7B explains why stocks are interesting
- 📈 **Visual dashboard** - Beautiful Streamlit interface with charts

**Example output:**
```
Found: RELIANCE.NS
Price: ₹2,847.50
RSI: 31.24 (Oversold)

AI Analysis:
INTERESTING: Yes
SIGNAL: Oversold
REASON: RSI at 28 indicates oversold condition with volume increase.
```

---

## ✨ Features (Sprint 1)

- ✅ **Market data collection** - Fetches stock prices from Yahoo Finance
- ✅ **Technical indicators** - RSI, MACD, Bollinger Bands (manual calculation)
- ✅ **AI Scanner Agent** - Uses free HuggingFace Mistral-7B model
- ✅ **Streamlit Dashboard** - Professional UI with interactive charts
- ✅ **100% Free** - No API costs

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+ (you have 3.11.9 ✅)
- pip
- Free HuggingFace account

### Installation

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd ai-trading-copilot
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Get your free HuggingFace token**
- Go to: https://huggingface.co/join
- Sign up (free, no credit card)
- Go to: https://huggingface.co/settings/tokens
- Create a new **Read** token
- Copy the token (starts with `hf_...`)

**4. Set up environment**
```bash
# Copy the example
cp .env.example .env

# Edit .env and add your token
# HUGGINGFACE_API_TOKEN=hf_your_token_here
```

**5. Run the dashboard**
```bash
streamlit run streamlit_app/app.py
```

Open http://localhost:8501 in your browser! 🎉

---

## 📁 Project Structure

```
ai-trading-copilot/
├── src/
│   ├── agents/
│   │   └── scanner_agent.py       # AI scanner using HuggingFace
│   ├── data_pipeline/
│   │   ├── collector.py            # Fetches stock data (Yahoo Finance)
│   │   └── indicators_simple.py    # Technical indicators (manual calc)
│   └── utils/
│       └── config.py               # Loads environment variables
├── streamlit_app/
│   └── app.py                      # Dashboard UI
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment template
└── README.md                       # This file
```

---

## 🎓 How It Works

### 1. **Data Collection**
```python
collector = MarketDataCollector()
data = collector.fetch_data("RELIANCE.NS", period="3mo")
# Returns: DataFrame with Open, High, Low, Close, Volume
```

### 2. **Calculate Indicators**
```python
calculator = SimpleTechnicalIndicators()
data_with_indicators = calculator.calculate_all(data)
# Adds: RSI, MACD, Bollinger Bands, Volume MA
```

### 3. **AI Analysis**
```python
scanner = MarketScanner()
results = scanner.scan(["RELIANCE.NS", "TCS.NS"])
# Returns: Stocks with AI explanations
```

### 4. **Display Results**
```bash
streamlit run streamlit_app/app.py
# Opens interactive dashboard
```

---

## 🧪 Testing

### Test Data Collector
```bash
python src/data_pipeline/collector.py
```

### Test Indicators
```bash
python src/data_pipeline/indicators_simple.py
```

### Test Scanner Agent
```bash
python src/agents/scanner_agent.py
```

---

## 📊 Technical Indicators Explained

### **RSI (Relative Strength Index)**
- **Range:** 0-100
- **Oversold:** < 30 (potential buy)
- **Overbought:** > 70 (potential sell)
- **Neutral:** 30-70

### **MACD (Moving Average Convergence Divergence)**
- **Positive MACD:** Bullish trend
- **Negative MACD:** Bearish trend
- **Crossover:** Potential signal

### **Bollinger Bands**
- **Upper Band:** Price + 2× std deviation
- **Lower Band:** Price - 2× std deviation
- **Price near upper:** Potentially overbought
- **Price near lower:** Potentially oversold

---

## 🤖 AI Model

**Using:** Mistral-7B-Instruct-v0.2

**Why this model?**
- ✅ 100% Free (no API costs)
- ✅ Open-source
- ✅ Good at technical analysis
- ✅ Fast responses
- ✅ No rate limits

**Trade-off vs paid models (like Claude):**
- Simpler reasoning
- Less nuanced analysis
- But totally free and works well!

---

## 🎯 Sprint 1 Deliverables

- [x] Market data collector (Yahoo Finance)
- [x] Technical indicator engine (RSI, MACD, BB)
- [x] Scanner agent (HuggingFace AI)
- [x] Streamlit dashboard with charts

**Status:** Sprint 1 Complete! ✅

---

## 🔜 Future Sprints

- **Sprint 2:** Multi-agent system with specialized strategies
- **Sprint 3:** Risk management engine
- **Sprint 4:** Paper trading integration
- **Sprint 5:** Trade journal & analytics
- **Sprint 6:** Production deployment

---

## ⚠️ Important Notes

### This is NOT financial advice
- Built as a learning project
- For educational purposes only
- Always do your own research
- Never invest more than you can afford to lose

### Data Limitations
- Yahoo Finance data may have delays
- Technical indicators are backward-looking
- Past performance ≠ future results

### AI Limitations
- AI can make mistakes
- Not as sophisticated as paid models
- Use as one input, not sole decision maker

---

## 🛠️ Tech Stack

**Language:** Python 3.11+

**Data & Analysis:**
- yfinance (market data)
- pandas (data processing)
- numpy (math operations)

**AI:**
- LangChain (agent framework)
- HuggingFace (free AI models)
- Mistral-7B (language model)

**UI:**
- Streamlit (dashboard)
- Plotly (interactive charts)

**Development:**
- python-dotenv (environment management)
- pytest (testing)

---

## 📚 Learning Resources

### Technical Analysis
- [Investopedia - Technical Analysis](https://www.investopedia.com/terms/t/technicalanalysis.asp)
- [RSI Explained](https://www.investopedia.com/terms/r/rsi.asp)
- [MACD Explained](https://www.investopedia.com/terms/m/macd.asp)

### AI & Agents
- [LangChain Documentation](https://python.langchain.com/)
- [HuggingFace Models](https://huggingface.co/models)

### Python & Data
- [Pandas Tutorial](https://pandas.pydata.org/docs/getting_started/intro_tutorials/)
- [Plotly Charts](https://plotly.com/python/)

---

## 🤝 Contributing

This is a learning project, but feedback welcome!

Found a bug? Have a suggestion? Open an issue!

---

## 📄 License

MIT License - Feel free to use for learning!

---

## 🙏 Acknowledgments

- **HuggingFace** - Free AI models
- **Yahoo Finance** - Free market data
- **Streamlit** - Amazing dashboard framework
- **Mistral AI** - Open-source Mistral-7B model

---

## 📧 Contact

**Built by:** Sohan (AI Engineering Learning Project)

---

⭐ **If you found this useful, give it a star!**

**Happy Trading! 🚀**

