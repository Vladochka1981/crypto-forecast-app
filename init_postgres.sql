-- ============================================================
-- init_postgres.sql — схема БД для PostgreSQL 16
-- (точная копия SQLite-схемы, с Postgres-типами)
-- ============================================================

-- ---------- klines ----------
CREATE TABLE IF NOT EXISTS klines (
    date   TEXT PRIMARY KEY,
    open   DOUBLE PRECISION NOT NULL,
    high   DOUBLE PRECISION NOT NULL,
    low    DOUBLE PRECISION NOT NULL,
    close  DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_klines_date ON klines(date);

-- ---------- fear_greed ----------
CREATE TABLE IF NOT EXISTS fear_greed (
    date              TEXT PRIMARY KEY,
    fear_greedy_value INTEGER
);
CREATE INDEX IF NOT EXISTS idx_fear_greed_date ON fear_greed(date);

-- ---------- onchain_metrics ----------
CREATE TABLE IF NOT EXISTS onchain_metrics (
    date               TEXT PRIMARY KEY,
    hash_rate          DOUBLE PRECISION,
    difficulty         DOUBLE PRECISION,
    n_tx               DOUBLE PRECISION,
    miners_revenue_btc DOUBLE PRECISION,
    trade_volume_btc   DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_onchain_date ON onchain_metrics(date);

-- ---------- macro_metrics ----------
CREATE TABLE IF NOT EXISTS macro_metrics (
    date        TEXT PRIMARY KEY,
    SP500       DOUBLE PRECISION,
    VIXCLS      DOUBLE PRECISION,
    DCOILWTICO  DOUBLE PRECISION,
    NASDAQQGLDI DOUBLE PRECISION,
    DFF         DOUBLE PRECISION,
    GEPUCURRENT DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_macro_date ON macro_metrics(date);

-- ---------- predictions ----------
-- (включая поля ACI: q_used, target_time, is_updated, alpha_t, version)
CREATE TABLE IF NOT EXISTS predictions (
    id          SERIAL PRIMARY KEY,
    created_at  TEXT NOT NULL,
    model       TEXT NOT NULL,
    horizon     INTEGER NOT NULL,
    anchor      DOUBLE PRECISION NOT NULL,
    point       DOUBLE PRECISION NOT NULL,
    lower       DOUBLE PRECISION,
    upper       DOUBLE PRECISION,
    is_stale    INTEGER NOT NULL DEFAULT 0,
    q_used      DOUBLE PRECISION,
    target_time TEXT,
    is_updated  INTEGER NOT NULL DEFAULT 0,
    alpha_t     DOUBLE PRECISION,
    version     TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_predictions_key
    ON predictions(model, horizon, anchor, created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_pred_pending ON predictions(is_updated, target_time);

-- ---------- model_versions ----------
CREATE TABLE IF NOT EXISTS model_versions (
    id                  SERIAL PRIMARY KEY,
    model_name          TEXT NOT NULL,
    version             TEXT NOT NULL,
    file_path           TEXT NOT NULL,
    is_active           INTEGER NOT NULL DEFAULT 0,
    trained_at          TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    metrics_snapshot    TEXT,
    dataset_fingerprint TEXT,
    UNIQUE (model_name, version)
);
CREATE INDEX IF NOT EXISTS idx_model_versions_active
    ON model_versions(model_name, is_active);

-- ---------- training_runs ----------
CREATE TABLE IF NOT EXISTS training_runs (
    id                  SERIAL PRIMARY KEY,
    model_version_id    INTEGER NOT NULL REFERENCES model_versions(id),
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    status              TEXT NOT NULL,
    hyperparameters     TEXT,
    metrics             TEXT,
    dataset_fingerprint TEXT,
    decision            TEXT,
    n_train             INTEGER,
    n_val               INTEGER,
    champion_run_id     TEXT
);
CREATE INDEX IF NOT EXISTS idx_training_runs_status
    ON training_runs(status, started_at DESC);