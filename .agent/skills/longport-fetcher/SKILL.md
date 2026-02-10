---
name: longport-fetcher
description: Use when the user asks to download or fetch HK/US stock quotes, candlesticks, or persist market data to disk for backtesting. Includes watchlist/single-symbol incremental download script, stock_data/ path rules, and MCP candlesticks quick lookup. Requires longport + mcp_config.json.
version: 1.5.0
---
# Longport Data Fetcher Skill

## When to use this skill

- User says "下载观察/自选 K 线"、"把某某标的日线落盘"、"更新 stock_data" 或 "fetch candles for backtesting".
- User needs to persist LongPort MCP data to local `stock_data/` with a fixed naming convention.
- User wants to use the recommended download script (watchlist groups, incremental, defaults "观察" + "esg 50").

Do **not** use for one-off quote lookups only (use MCP tools directly). Use when **persistence** or **batch download** is required.

## Data Storage Rules

**CRITICAL: All fetched data MUST be saved to the `stock_data` directory.**

- **Root Directory**: Ensure `stock_data/` exists in the project root.
- **Naming Convention**:
    - Candlesticks: `stock_data/{symbol}_{period}.csv` (e.g., `stock_data/AAPL.US_day.csv`)
    - Quotes/Info: `stock_data/quotes_{timestamp}.json`
- **Format**: CSV is preferred for tabular data (candlesticks), JSON for raw metadata.

## Tools & Capabilities

### 1. Robust Download Script (Watchlist, Incremental, Intraday)
**RECOMMENDED:** The provided Python script handles efficient data downloading.

**Script Location:** `.agent/skills/longport-fetcher/scripts/download_candles.py`

**Usage:**
```bash
python3 .agent/skills/longport-fetcher/scripts/download_candles.py [OPTIONS]
```

**Defaults:**
If no `--symbol` or `--watchlist` is provided, the script automatically downloads data for the following watchlist groups:
- **"观察"**
- **"esg 50"**

**Arguments:**
- `symbol` (positional, optional): Stock symbol.
- `--watchlist <GROUPS>`: Comma-separated watchlist group names. Default: "观察,esg 50".
- `--period`: `day` (default).
- `--count`: `1000` (default).

**Examples:**
```bash
# 1. Default: Download "观察" and "esg 50" groups
python3 .agent/skills/longport-fetcher/scripts/download_candles.py

# 2. Specific Group
python3 .agent/skills/longport-fetcher/scripts/download_candles.py --watchlist "HK Holdings"

# 3. Single Stock
python3 .agent/skills/longport-fetcher/scripts/download_candles.py --symbol 700.HK
```

### 2. Historical Candlesticks (MCP Tool)
Use `mcp_longport-mcp_candlesticks` for quick lookups.

## Setup Requirements

```bash
pip install longport
```
The script requires `mcp_config.json` for credentials.

---

## Scripts & resources index (Level 3, load when needed)

| Purpose | Path | Notes |
|---------|------|--------|
| Watchlist / single-symbol download | `.agent/skills/longport-fetcher/scripts/download_candles.py` | Default watchlist "观察", "esg 50"; `--watchlist`, `--period`, `--count` |
| Quick candlestick lookup | MCP `mcp_longport-mcp_candlesticks` | No persistence; use script for saving to disk |

**Storage:** All fetched data must go under `stock_data/`. Candlesticks: `stock_data/{symbol}_{period}.csv`; quotes/info: `stock_data/quotes_{timestamp}.json`. This skill defines the process and paths; LongPort MCP provides the data connection.
