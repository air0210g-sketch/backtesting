"""
道氏理论 123 法则与 2B 法则的量化回测。
入场/出场逻辑统一由 backtesting.entry_strategies 提供，便于组合（如 entry_2b & entry_123）。

运行（项目根目录）：
  .venv/bin/python scripts/run_dow_123_2b.py [symbol] [--entry ma|123|2b|both]
  不传 symbol 默认 0700.HK。--entry：ma=均线金叉，123=123 入场，2b=2B 入场，both=123 或 2B 任一。
"""
import os
import sys
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backtesting.data_loader import load_data
from backtesting.entry_strategies import (
    entry_123,
    entry_2b,
    entry_ma,
    exit_123,
    exit_2b,
)
import vectorbt as vbt

PIVOT_ORDER = 5
LOOKBACK = 135
MA_ENTRY = 20

DATA_DIR = os.path.join(_REPO_ROOT, "stock_data")


def run_dow_123_2b(
    symbol: str = "0700.HK",
    days: int = 365,
    entry_mode: str = "ma",
):
    """
    加载数据，用 entry_strategies 计算入场/出场并回测。
    entry_mode: ma=均线金叉, 123=123 入场, 2b=2B 入场, both=123 或 2B 任一。
    """
    data = load_data(DATA_DIR, symbols=[symbol])
    if not data or symbol not in data:
        print(f"无数据: {symbol}")
        return None

    df = data[symbol].tail(days)
    close_series = df["close"]
    close_df = close_series.to_frame(symbol) if isinstance(close_series, pd.Series) else close_series
    high_series = df["high"]
    high_df = high_series.to_frame(symbol) if isinstance(high_series, pd.Series) else high_series
    low_series = df["low"]
    low_df = low_series.to_frame(symbol) if isinstance(low_series, pd.Series) else low_series

    if entry_mode == "ma":
        entries = entry_ma(close_df, period=MA_ENTRY)
        entry_label = f"收盘上穿 MA{MA_ENTRY}"
    elif entry_mode == "123":
        entries = entry_123(
            high_df, low_df, close_df,
            pivot_order=PIVOT_ORDER, lookback=LOOKBACK,
        )
        entry_label = "123 法则（下降趋势反转）"
    elif entry_mode == "2b":
        entries = entry_2b(
            high_df, low_df, close_df,
            pivot_order=PIVOT_ORDER, lookback=LOOKBACK,
        )
        entry_label = "2B 法则（假跌破）"
    else:  # both
        e123 = entry_123(
            high_df, low_df, close_df,
            pivot_order=PIVOT_ORDER, lookback=LOOKBACK,
        )
        e2b = entry_2b(
            high_df, low_df, close_df,
            pivot_order=PIVOT_ORDER, lookback=LOOKBACK,
        )
        entries = e123 | e2b
        entry_label = "123 或 2B"

    exits = exit_123(
        high_df, low_df, close_df,
        pivot_order=PIVOT_ORDER, lookback=LOOKBACK,
    ) | exit_2b(
        high_df, low_df, close_df,
        pivot_order=PIVOT_ORDER, lookback=LOOKBACK,
    )

    pf = vbt.Portfolio.from_signals(
        close=close_df,
        entries=entries,
        exits=exits,
        init_cash=1_000_000,
        fees=0.0003,
        slippage=0.0005,
        freq="1D",
    )

    tr = pf.total_return()
    md = pf.max_drawdown()
    n_trades = int(pf.trades.count().sum())
    ret_val = tr.iloc[-1] if hasattr(tr, "iloc") else float(tr)
    md_val = md.iloc[-1] if hasattr(md, "iloc") else float(md)

    print(f"[道氏 123/2B] 标的: {symbol} | 入场: {entry_label} | 出场: 123 或 2B")
    print(f"总收益率: {ret_val:.4f}")
    print(f"最大回撤: {md_val:.4f}")
    print(f"交易次数: {n_trades}")
    return pf


if __name__ == "__main__":
    entry_mode = "ma"
    for i, a in enumerate(sys.argv[1:], start=1):
        if a.startswith("--entry="):
            entry_mode = a.split("=", 1)[1].strip().lower()
            break
        if a == "--entry" and i < len(sys.argv) - 1:
            entry_mode = sys.argv[i + 1].strip().lower()
            break
    if entry_mode not in ("ma", "123", "2b", "both"):
        entry_mode = "ma"
    symbol = "0700.HK"
    skip_next = False
    for a in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if a == "--entry":
            skip_next = True
            continue
        if a.startswith("--"):
            continue
        symbol = a
        break
    run_dow_123_2b(symbol, entry_mode=entry_mode)
