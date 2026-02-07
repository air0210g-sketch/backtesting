---
name: longport-fetcher
description: Fetch stock market data (quotes, candlesticks, info) using the Longport MCP tools and persist them to disk.
version: 1.5.0
---

# Longport Data Fetcher Skill

This skill provides a standardized way to fetch market data using the `longport-mcp` tools and **save them locally** for backtesting.

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
