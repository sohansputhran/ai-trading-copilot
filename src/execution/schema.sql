-- trades table - designed for Sprint 5 analytics queries
--
-- Key analytics queries we're designing for:
--   1. Win rate:      COUNT(CASE WHEN pnl > 0 THEN 1 END) / COUNT(*)
--   2. Sharpe ratio:  AVG(pnl_pct) / STDDEV(pnl_pct) × sqrt(252)
--   3. R-multiple:    AVG(r_multiple) WHERE status IN ('CLOSED', 'STOPPED_OUT')
--   4. Max drawdown:  Running equity curve from cumulative pnl
--   5. By strategy:   GROUP BY strategy
--   6. Time analysis: GROUP BY DATE(fill_timestamp)
--
-- Every field that appears in a WHERE or GROUP BY has an index.

CREATE TABLE IF NOT EXISTS trades (
    -- Identity
    order_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol              VARCHAR(20)  NOT NULL,
    side                VARCHAR(4)   NOT NULL CHECK (side IN ('BUY', 'SELL')),

    -- Sizing & risk
    shares              INTEGER      NOT NULL,
    requested_price     NUMERIC(12, 2),
    stop_loss           NUMERIC(12, 2),
    take_profit         NUMERIC(12, 2),
    capital_at_risk     NUMERIC(12, 2),

    -- Context (for attribution and explainability)
    strategy            VARCHAR(50),
    agent_reasoning     TEXT,
    confidence          NUMERIC(5, 4),

    -- Lifecycle
    status              VARCHAR(15)  NOT NULL
                            CHECK (status IN ('PENDING','FILLED','REJECTED','CANCELLED','CLOSED','STOPPED_OUT')),
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    -- Fill details
    fill_price          NUMERIC(12, 2),
    fill_timestamp      TIMESTAMPTZ,
    slippage            NUMERIC(10, 4),

    -- Exit details
    exit_price          NUMERIC(12, 2),
    exit_timestamp      TIMESTAMPTZ,
    exit_reason         VARCHAR(20)  CHECK (exit_reason IN ('take_profit', 'stop_loss', 'manual', NULL)),

    -- Performance (calculated on close)
    pnl                 NUMERIC(12, 2),
    pnl_pct             NUMERIC(8, 4),
    r_multiple          NUMERIC(8, 4)
);

-- Indexes for Sprint 5 analytics queries

-- Most queries filter by status (open positions, closed trades)
CREATE INDEX IF NOT EXISTS idx_trades_status      ON trades (status);

-- Time-bucketed analysis (daily P&L, equity curve)
CREATE INDEX IF NOT EXISTS idx_trades_fill_ts     ON trades (fill_timestamp);

-- Per-symbol P&L attribution
CREATE INDEX IF NOT EXISTS idx_trades_symbol      ON trades (symbol);

-- Per-strategy performance comparison
CREATE INDEX IF NOT EXISTS idx_trades_strategy    ON trades (strategy);

-- Convenience view: closed trades only
CREATE OR REPLACE VIEW closed_trades AS
    SELECT *
    FROM   trades
    WHERE  status IN ('CLOSED', 'STOPPED_OUT');

-- Convenience view: running equity curve
CREATE OR REPLACE VIEW equity_curve AS
    SELECT
        fill_timestamp::DATE            AS trade_date,
        SUM(pnl)                        AS daily_pnl,
        SUM(SUM(pnl)) OVER (
            ORDER BY fill_timestamp::DATE
        )                               AS cumulative_pnl
    FROM   closed_trades
    GROUP  BY fill_timestamp::DATE
    ORDER  BY trade_date;
