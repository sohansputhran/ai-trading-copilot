# Key Technical Decisions

> **3-minute read**: Why I made specific architectural and implementation choices

---

## Decision Log

### 1. **Multi-Agent System over Single Agent**
**Date**: Sprint 2  
**Decision**: Use 3 specialized agents instead of 1 generalist agent  
**Rationale**:
- **Reduces false positives**: Consensus voting (67% threshold) filters weak signals
- **Specialization**: Each agent focuses on one strategy (technical, momentum, breakout)
- **Explainability**: Can show which agents agreed/disagreed and why
- **Real-world parallel**: Mirrors how trading desks work (multiple analysts, not one oracle)

**Trade-off**: More API calls (3x), but better decision quality outweighs cost (and it's free with HuggingFace)

**Result**: Successfully flagged high-potential stocks while avoiding noise (60% win rate in paper trading)

---

### 2. **Repository Pattern for Data Persistence**
**Date**: Sprint 5  
**Decision**: Abstract data access behind `TradeRepository` ABC  
**Rationale**:
- **Swappable backends**: Built with SQLite, migrated to PostgreSQL in 30 minutes
- **Zero application code changes**: Only swapped the repository class
- **Testability**: Easy to create in-memory test repositories
- **Future-proof**: Can add MongoDB, DynamoDB, etc. without touching business logic

**Implementation**:
```python
# src/journal/repository.py
class TradeRepository(ABC):
    @abstractmethod
    def save_order(self, order: Order) -> None: ...
    
    @abstractmethod
    def get_open_orders(self) -> List[Order]: ...

# src/journal/sqlite_repo.py
class SQLiteRepository(TradeRepository):
    # SQLite-specific implementation

# src/journal/postgresql_repo.py  
class PostgreSQLRepository(TradeRepository):
    # PostgreSQL-specific implementation
```

**Trade-off**: More upfront code (ABCs, multiple classes), but saved 10+ hours during migration

---

### 3. **Manual Indicator Implementation (No pandas-ta)**
**Date**: Sprint 1  
**Decision**: Manually implement RSI, MACD, etc. instead of using pandas-ta library  
**Rationale**:
- **Environment compatibility**: pandas-ta requires Python 3.12+, my environment is 3.11
- **Learning**: Understanding the math deepens domain knowledge
- **Control**: Can optimize and debug easily
- **No black box**: Know exactly what's being calculated

**Example** (RSI calculation):
```python
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
```

**Trade-off**: More code to write/maintain, but eliminated dependency issues and learned deeply

---

### 4. **Read Model Pattern for Analytics**
**Date**: Sprint 5  
**Decision**: Create `ClosedTrade` read model with derived fields instead of storing everything  
**Rationale**:
- **No schema migration needed**: Added `hold_days` and `r_multiple` without altering DB
- **Single source of truth**: Calculations derive from stored data, not duplicated
- **Simpler testing**: Generate read models on-the-fly in tests

**Implementation**:
```python
@dataclass
class ClosedTrade:
    # Stored fields
    symbol: str
    entry_price: Decimal
    exit_price: Decimal
    # ...
    
    # Derived fields (computed at read time)
    @property
    def hold_days(self) -> int:
        return (self.exit_timestamp - self.entry_timestamp).days
    
    @property
    def r_multiple(self) -> Decimal:
        risk = self.entry_price - self.stop_loss
        return self.pnl / risk if risk > 0 else Decimal("0")
```

**Trade-off**: Slight performance cost (re-computation), but negligible for 1000s of trades

---

### 5. **Stateless Analytics Engine**
**Date**: Sprint 5  
**Decision**: `PerformanceEngine` has only pure static methods, no instance state  
**Rationale**:
- **Testability**: Easy to test (just call methods with fixtures)
- **Thread-safe**: No shared state = no race conditions
- **Clarity**: Clear that engine doesn't modify data, just computes metrics
- **Functional style**: Easier to reason about (input → output, no side effects)

**Example**:
```python
class PerformanceEngine:
    @staticmethod
    def calculate_win_rate(trades: List[ClosedTrade]) -> Decimal:
        if not trades:
            return Decimal("0")
        wins = sum(1 for t in trades if t.pnl > 0)
        return Decimal(wins) / Decimal(len(trades))
    
    @staticmethod
    def calculate_sharpe_ratio(trades: List[ClosedTrade]) -> Decimal:
        # Pure function - no state needed
        ...
```

**Trade-off**: None really — this is just cleaner design

---

### 6. **SQLite First, PostgreSQL Later**
**Date**: Sprint 1 → Sprint 6 transition  
**Decision**: Build entire system on SQLite, migrate to PostgreSQL in Sprint 6  
**Rationale**:
- **Faster development**: Zero setup, file-based, portable
- **Sufficient for learning**: Handled 1000+ trades without issues
- **Easy testing**: In-memory SQLite for unit tests (`:memory:`)
- **Production path**: Repository pattern made migration trivial

**Migration approach**:
```python
# Development
repo = SQLiteRepository("trades.db")

# Production
repo = PostgreSQLRepository(os.getenv("DATABASE_URL"))

# Application code (unchanged)
repo.save_order(order)  # Works with both!
```

**Trade-off**: Had to plan for PostgreSQL from the start (schema, queries), but worth it

---

### 7. **HuggingFace over OpenAI/Anthropic**
**Date**: Sprint 1  
**Decision**: Use free HuggingFace Inference API instead of paid LLM APIs  
**Rationale**:
- **100% free**: No API costs (important for portfolio project)
- **Multi-model fallback**: Chain of 4 models (Llama-3 → Flan-T5 → Zephyr → rules)
- **Good enough**: For signal classification, free models work well with good prompts
- **Open-source**: Aligns with project values

**Prompt engineering to compensate**:
```python
system_prompt = """
You are a strict stock analyst. 
~70-80% of stocks should be classified as INTERESTING: No.
Only flag stocks with STRONG signals.
"""
```

**Trade-off**: Less nuanced reasoning than GPT-4/Claude, but sufficient for this use case

---

### 8. **Consensus Threshold of 67%**
**Date**: Sprint 2  
**Decision**: Require 2 out of 3 agents to agree (67%) before flagging a stock  
**Rationale**:
- **Tested alternatives**:
  - 100% (all 3 agents) → Too strict, missed opportunities
  - 50% (any 2 agents) → Too lenient initially, but 67% was the sweet spot
- **Balanced**: Filters noise while catching genuine signals
- **Tunable**: Can adjust based on market conditions

**Result**: Achieved 60% win rate in paper trading (vs 45% with single agent)

---

### 9. **Risk Gates in Aggregator**
**Date**: Sprint 3  
**Decision**: Add ADX and volume filters AFTER multi-agent consensus  
**Rationale**:
- **Bug fix**: Initially all stocks returned HOLD due to strict filters
- **Layered approach**: 
  1. Multi-agent consensus (intent)
  2. Technical gates (quality check)
- **Tunable thresholds**: ADX ≥ 15 (was 20), Volume ≥ 1.2x (was 1.5x)

**Before vs After**:
```python
# Before (too strict)
if adx >= 20 and volume_ratio >= 1.5 and consensus >= 0.67:
    return "BUY"  # Almost nothing passed
    
# After (relaxed gates)
if consensus >= 0.67:  # First check intent
    if adx >= 15 and volume_ratio >= 1.2:  # Then validate quality
        return "BUY"  # Much better hit rate
```

---

### 10. **Portfolio Hydration on App Startup**
**Date**: Sprint 5 (Bug Fix)  
**Decision**: Re-sync in-memory `PortfolioRisk` from SQLite on Streamlit app reload  
**Rationale**:
- **Problem**: Streamlit reruns on every interaction, losing in-memory state
- **Solution**: Check if portfolio already hydrated (`st.session_state`) before re-initializing
- **Guard key**: `portfolio_hydrated` flag prevents multiple hydrations

**Implementation**:
```python
if "portfolio_hydrated" not in st.session_state:
    open_orders = repo.get_open_orders()
    for order in open_orders:
        portfolio.add_position(...)
    st.session_state["portfolio_hydrated"] = True
```

**Trade-off**: Small startup cost (50ms), but ensures consistency

---

### 11. **Test Fixtures over Mocking**
**Date**: Sprint 5  
**Decision**: Use synthetic dataclasses in tests instead of mocking repository  
**Rationale**:
- **Simpler**: No mock setup/teardown
- **Portable**: Tests run anywhere without external dependencies
- **Readable**: Clear what data looks like
- **Fast**: No database I/O in unit tests

**Example**:
```python
# tests/test_analytics.py
def test_win_rate():
    trades = [
        ClosedTrade(pnl=Decimal("100"), ...),  # Win
        ClosedTrade(pnl=Decimal("-50"), ...),  # Loss
        ClosedTrade(pnl=Decimal("200"), ...),  # Win
    ]
    win_rate = PerformanceEngine.calculate_win_rate(trades)
    assert win_rate == Decimal("0.6666666667")  # 2/3
```

**Trade-off**: Have to manually create fixtures, but clearer and faster than mocking

---

### 12. **CI/CD with Separate Requirements**
**Date**: Sprint 5  
**Decision**: Create `requirements-ci.txt` excluding torch/transformers  
**Rationale**:
- **Speed**: CI runs in <30s (vs 2+ minutes with full deps)
- **No HuggingFace in tests**: Tests use synthetic fixtures, don't need models
- **Practical**: Balance between coverage and CI cost

**Structure**:
```
requirements.txt         # Full deps (local dev + deployment)
requirements-ci.txt      # Minimal deps (linting + testing only)
```

**Trade-off**: Maintain two files, but CI is way faster

---

## Decision Themes

### 1. **Start Simple, Plan for Scale**
- SQLite → PostgreSQL path
- Single agent → Multi-agent evolution
- In-memory → Persistent storage

### 2. **Prefer Patterns over Hacks**
- Repository, Strategy, Read Model
- ABC contracts everywhere
- Stateless engines

### 3. **Test-Driven Thinking**
- Synthetic fixtures (no mocks)
- Pure functions (easy to test)
- Zero external deps in tests

### 4. **Balance Learning and Pragmatism**
- Manual indicators (learn deeply)
- Free HuggingFace (practical for portfolio)
- Incremental sprints (steady progress)

---

## What I'd Do Differently

### If Starting Over:
1. **PostgreSQL from Day 1**: Would save migration effort (but SQLite was faster to learn)
2. **More integration tests**: Currently only unit tests (30), need E2E coverage
3. **Better error messages**: Some errors are too generic (need more context)

### If Going to Production:
1. **Redis caching**: Multi-process safe (current is in-memory dict)
2. **Message queue**: Async scans (RabbitMQ/Celery)
3. **Monitoring**: Prometheus + Grafana for observability
4. **Rate limiting**: Prevent API abuse on public endpoints

---

## Lessons Learned

### Technical:
- **Repository pattern is worth it**: Saved hours during migration
- **ABCs enforce discipline**: Caught interface mismatches early
- **Stateless > Stateful**: Easier to test and reason about
- **Free LLMs work**: With good prompts and fallbacks

### Process:
- **Sprint structure works**: 6 sprints, each 2 weeks, clear milestones
- **Document decisions**: This file would've saved debugging time
- **Test as you go**: Catching bugs early is 10x easier
- **Incremental delivery**: Better than big-bang approach

---

**Key Insight**: Good architecture isn't about using fancy patterns — it's about making future changes easy. The repository pattern, ABCs, and stateless engines all proved their worth when requirements evolved.

---

**Last Updated**: Sprint 5 Complete (January 2025)
