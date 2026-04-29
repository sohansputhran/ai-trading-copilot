# System Architecture

> **2-minute read**: How the AI Trading Copilot is designed

---

## High-Level Design

```
┌─────────────────────────────────────────────────┐
│         STREAMLIT DASHBOARD (UI)                │
│  Market Scanner | Agent Reasoning | Analytics   │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│      MULTI-AGENT ORCHESTRATION (LangGraph)      │
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │Technical │  │Momentum  │  │Breakout  │     │
│  │ Agent    │  │ Agent    │  │ Agent    │     │
│  └──────────┘  └──────────┘  └──────────┘     │
│         │            │            │            │
│         └────────────┴────────────┘            │
│                   ▼                            │
│         Multi-Agent Aggregator                 │
│         (67% consensus required)               │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│              DATA LAYER                         │
│                                                 │
│  Market Data    Risk Engine    Trade Journal   │
│  (Yahoo + NSE)  (Kelly + Gates) (SQLite/PG)    │
└─────────────────────────────────────────────────┘
```

---

## Core Components

### 1. **Multi-Agent System** (LangGraph)
- **3 specialized agents**: Each analyzes stocks from a different strategy lens
- **Consensus voting**: Requires ≥67% agreement before flagging a stock
- **Explainable decisions**: Every recommendation includes reasoning chains

**Why multi-agent?**
- More robust than single-agent (reduces false positives)
- Mirrors real trading desk structure (multiple analysts)
- Easily extensible (add more agents without breaking existing ones)

### 2. **Technical Analysis Engine**
- **10+ indicators**: RSI, MACD, Bollinger Bands, EMA, ADX, ATR, Volume
- **Manual implementations**: No pandas-ta dependency (environment compatibility)
- **Efficient computation**: Vectorized operations with NumPy

**Key indicators**:
```python
RSI < 30        → Oversold signal
MACD crossover  → Momentum shift
BB breakout     → Volatility expansion
Volume > 1.5x   → Institutional interest
```

### 3. **Risk Management**
- **Position sizing**: Kelly Criterion (with safety caps)
- **Portfolio constraints**:
  - Max 10% capital per trade
  - Max 5 open positions
  - Max 30% sector exposure
- **Pre-trade gates**:
  - ADX ≥ 15 (trend strength)
  - Volume ≥ 1.2x average
  - Multi-agent consensus

### 4. **Data Pipeline**
- **Source**: Yahoo Finance (yfinance) for Indian stocks
- **Caching**: 5-minute TTL to avoid rate limits
- **Validation**: Data completeness checks before analysis

### 5. **Trade Journal & Analytics**
- **Repository pattern**: ABC with SQLite/PostgreSQL implementations
- **Read models**: `ClosedTrade` with derived fields (hold_days, R-multiple)
- **Performance metrics**:
  - Win rate, profit factor, expectancy
  - Sharpe ratio (√252 annualized)
  - Sortino ratio, max drawdown

---

## Design Patterns Used

| Pattern | Where | Why |
|---------|-------|-----|
| **Repository** | Trade storage | Swappable DB (SQLite → PostgreSQL) |
| **Strategy** | Agent system | Each agent = different strategy |
| **Abstract Base Class** | BrokerInterface, TradeRepository | Enforce contracts |
| **Read Model** | ClosedTrade | Derived fields computed at read time |
| **Stateless Engine** | PerformanceEngine | Pure static methods, no state |

---

## Technology Choices

### Python 3.11+
- Rich ML/finance ecosystem
- Native async support (FastAPI ready)
- Type hints for maintainability

### LangGraph (Multi-Agent Framework)
- **vs LangChain**: Better state management for orchestration
- **vs CrewAI**: More control over agent interactions
- Built-in visualization for debugging

### HuggingFace Inference API
- **100% free** (no OpenAI/Anthropic costs)
- Multi-model fallback chain (Llama-3 → Flan-T5 → Zephyr)
- `InferenceClient` for latest API compatibility

### Streamlit (UI)
- **vs Flask/Django**: Faster iteration for dashboards
- **vs React**: No separate frontend stack
- Portfolio-ready with custom CSS

### SQLite → PostgreSQL Path
- **SQLite first**: Faster development, zero setup
- **PostgreSQL later**: Production-ready, better analytics
- **Repository pattern**: Switched DBs in 30 minutes (zero app code changes)

---

## Data Flow Example

### Stock Analysis Workflow
```
1. User clicks "Scan Market"
   ↓
2. System fetches 20 stocks (Yahoo Finance)
   ↓
3. Calculate indicators (RSI, MACD, etc.)
   ↓
4. Each agent analyzes:
   - Technical Agent  → "BUY (RSI oversold)"
   - Momentum Agent   → "HOLD (weak trend)"
   - Breakout Agent   → "BUY (volume spike)"
   ↓
5. Aggregator: 2/3 = 67% → BUY signal
   ↓
6. Risk gates check:
   - ADX ≥ 15? ✓
   - Volume ≥ 1.2x? ✓
   - Portfolio capacity? ✓
   ↓
7. Display result with reasoning
```

---

## Scalability Considerations

### Current (Sprint 5)
- ✅ Handles 50 stocks/scan
- ✅ 1000+ trades in journal
- ✅ Sub-second indicator calculations

### Future (Production)
- 🔄 Redis caching layer (multi-process)
- 🔄 Message queue for async scans (RabbitMQ)
- 🔄 Horizontal scaling (containerized agents)

---

## Security & Safety

### API Keys
- Environment variables only (`.env`)
- Never committed to Git (`.gitignore`)

### Risk Controls
- **Hard limits**: Cannot be bypassed in code
- **Pre-trade validation**: All checks must pass
- **Paper trading first**: No real money until proven

### Data Validation
- Type checking (Pydantic models)
- Input sanitization (SQL injection prevention)
- Error handling with fallbacks

---

## Testing Strategy

### Unit Tests (30 tests)
- **Indicators**: Hand-verified expected values
- **Risk engine**: Edge cases (zero capital, max positions)
- **Analytics**: Synthetic trade fixtures
- **Zero external deps**: No API calls in tests

### Integration Tests (Future)
- End-to-end scan workflow
- Database migrations
- Broker API mocking

---

## What Makes This Production-Grade?

| Aspect | Implementation |
|--------|---------------|
| **Modularity** | ABC contracts, clear interfaces |
| **Testability** | 30 unit tests, synthetic fixtures |
| **Observability** | Structured logging, error tracking |
| **Reliability** | Fallback chains (AI + rule-based) |
| **Maintainability** | Type hints, docstrings, patterns |
| **Scalability** | Stateless engines, repository pattern |

---

## Evolution Path

### Sprint 1-2: Foundation
- Single scanner agent → Multi-agent system
- Rule-based → AI-enhanced decisions

### Sprint 3-4: Robustness
- Basic risk checks → Kelly Criterion + gates
- Mock broker → Real paper trading

### Sprint 5: Analytics
- In-memory state → Persistent journal
- Manual calculation → Automated metrics

### Sprint 6: Production (Planned)
- SQLite → PostgreSQL migration
- Local → Docker deployment
- Manual testing → CI/CD pipeline

---

## Performance Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Scan time (20 stocks) | 8-12s | <5s |
| Indicator calculation | <1s | <500ms |
| Agent decision | 2-3s | <2s |
| Database query | <50ms | <100ms |
| Test coverage | 30 tests | 80%+ |

---

## Key Takeaways for Recruiters

1. **Clean architecture** - Not spaghetti code, uses proven patterns
2. **Multi-agent AI** - Beyond basic LLM wrappers, real orchestration
3. **Production thinking** - Tests, error handling, swappable components
4. **Domain expertise** - Implemented complex finance concepts correctly
5. **Incremental delivery** - 6 sprints, each builds on the last

---

**Built over 6 sprints (12 weeks) as a portfolio project to master AI Engineering.**
