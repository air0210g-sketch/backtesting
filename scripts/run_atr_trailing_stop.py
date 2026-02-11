"""
使用 VectorBT 实现「ATR 倍数移动止损」：
1) from_signals + adjust_sl_func_nb：每 bar 止损 = 当前 bar 的 high − ATR×mult。
2) from_order_func：严格 trailing = 自入场以来最高价 peak − ATR×mult，跌破即止损。

参考：https://vectorbt.dev/api/portfolio/base/#vectorbt.portfolio.base.Portfolio.from_signals

运行（项目根目录）：
  .venv/bin/python scripts/run_atr_trailing_stop.py [symbol] [--strict | --combined | --compare]
  不传 symbol 则用 0700.HK。
  --strict：严格 trailing = peak − 2×ATR，触发一次全平。
  --combined：组合止损，满足其一即触发 = ① 回撤≥20% ② peak−2×ATR；第1、2次平 30%，第3次清完。
  --compare：依次跑三种策略并输出对比表。

策略对比（优劣简述，以 0700.HK 近 365 日为例）：
  默认(每 bar high−2×ATR)：收益最高、回撤中等、交易少；止损参考“当日最高”偏松，易拿住趋势。
  严格(peak−2×ATR 一次全平)：收益最低、回撤最小、交易略多；止损最紧，保护强但易被震出。
  组合(20% 或 2×ATR + 分步平仓)：收益与回撤居中、交易最多；双条件+分步减仓，兼顾保护与弹性。
"""
import os
import sys
import numpy as np
import pandas as pd
from numba import njit

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backtesting.data_loader import load_data
from backtesting.indicators import calc_atr
import vectorbt as vbt
from vectorbt.portfolio import nb
from vectorbt.portfolio.enums import Direction, NoOrder

# ATR 倍数（回撤 ATR*ATR_MULT 时止损）
ATR_MULT = 2.0
ATR_PERIOD = 14
# 组合止损：移动止损比例（自高点回撤超过该比例即止损）
SL_PCT = 0.20  # 20%
# 每次触发止损/出场时平仓比例（第 1、2 次触发用此比例，第 3 次清完）
CLOSE_PCT = 0.30  # 30%
CLOSE_ALL_ON_NTH = 3  # 第 N 次触发时全平
MIN_POSITION = 1e-9  # 剩余仓位小于此视为清仓并重置


@njit
def adjust_sl_atr_nb(c, atr_arr, high_arr, atr_mult):
    """
    每 bar 将止损设为：当前 bar 最高价 - atr_mult * ATR。
    即止损比例 = (atr_mult * ATR) / high，trailing=True。
    AdjustSLContext 无 flex_2d，故用 2D 下标直接取值。
    """
    i, col = c.i, c.col
    atr_val = atr_arr[i, col]
    high_val = high_arr[i, col]
    if high_val <= 0 or atr_val <= 0 or np.isnan(atr_val) or np.isnan(high_val):
        return c.curr_stop, c.curr_trail
    pct = (atr_mult * atr_val) / high_val
    if pct <= 0:
        return c.curr_stop, c.curr_trail
    if pct >= 1:
        pct = 0.99
    return pct, True  # True = 使用 trailing（参考价为高点）


# ---------- 严格 trailing：peak（自入场以来最高价）− atr_mult×ATR ----------
# 使用 from_order_func，在 order_func 内维护每列的 high_since_entry（入场后每 bar 用 max(peak, high) 更新），
# 当 low <= peak - atr_mult*ATR 时平仓；平仓后将 peak 置为 NaN，下次入场再初始化。
@njit
def order_func_peak_atr_nb(
    c,
    high_since_entry,
    entries_arr,
    exits_arr,
    high_arr,
    low_arr,
    atr_arr,
    close_arr,
    atr_mult,
):
    """
    入场：信号为 True 且当前无仓位则买入，并设 high_since_entry[col] = 当前 high。
    持仓中：peak = max(peak, 当前 high)；若 low <= peak - atr_mult*ATR 则平仓（止损）。
    出场：exits 为 True 则平仓。
    """
    i, col = c.i, c.col
    high_now = high_arr[i, col]
    low_now = low_arr[i, col]
    atr_now = atr_arr[i, col]
    close_now = close_arr[i, col]
    entry_now = entries_arr[i, col]
    exit_now = exits_arr[i, col]

    if entry_now and c.position_now == 0:
        high_since_entry[col] = high_now
        return nb.order_nb(
            size=np.inf,
            price=close_now,
            direction=Direction.LongOnly,
            fees=0.0003,
            slippage=0.0005,
        )

    if c.position_now > 0:
        if np.isnan(high_since_entry[col]):
            high_since_entry[col] = high_now
        elif high_now > high_since_entry[col]:
            high_since_entry[col] = high_now

        peak = high_since_entry[col]
        stop_level = peak - atr_mult * atr_now
        if atr_now > 0 and not np.isnan(atr_now) and low_now <= stop_level:
            high_since_entry[col] = np.nan  # 平仓后重置，下次入场再设
            return nb.close_position_nb(price=close_now, fees=0.0003, slippage=0.0005)

        if exit_now:
            high_since_entry[col] = np.nan
            return nb.close_position_nb(price=close_now, fees=0.0003, slippage=0.0005)

    return NoOrder


# ---------- 组合止损：满足其一即触发；第 1、2 次平仓 close_pct，第 N 次全平 ----------
# 1) 移动止损超过 20%  2) peak − 2*ATR
# 触发计数：第 1、2 次各平 30%，第 3 次清完并重置。
@njit
def order_func_combined_nb(
    c,
    high_since_entry,
    close_count,
    entries_arr,
    exits_arr,
    high_arr,
    low_arr,
    atr_arr,
    close_arr,
    sl_pct,
    atr_mult,
    close_pct,
    close_all_on_nth,
    min_position,
):
    """入场后维护 peak；触发时第 1、2 次平 close_pct（30%），第 close_all_on_nth 次（第 3 次）全平。"""
    i, col = c.i, c.col
    high_now = high_arr[i, col]
    low_now = low_arr[i, col]
    atr_now = atr_arr[i, col]
    close_now = close_arr[i, col]
    entry_now = entries_arr[i, col]
    exit_now = exits_arr[i, col]

    if entry_now and c.position_now == 0:
        high_since_entry[col] = high_now
        close_count[col] = 0
        return nb.order_nb(
            size=np.inf,
            price=close_now,
            direction=Direction.LongOnly,
            fees=0.0003,
            slippage=0.0005,
        )

    if c.position_now > 0:
        if np.isnan(high_since_entry[col]):
            high_since_entry[col] = high_now
        elif high_now > high_since_entry[col]:
            high_since_entry[col] = high_now

        peak = high_since_entry[col]
        stop_level_pct = peak * (1.0 - sl_pct)
        stop_level_atr = peak - atr_mult * atr_now if (atr_now > 0 and not np.isnan(atr_now)) else np.nan

        hit_pct = low_now <= stop_level_pct
        hit_atr = not np.isnan(stop_level_atr) and low_now <= stop_level_atr
        if hit_pct or hit_atr or exit_now:
            close_count[col] += 1
            if close_count[col] >= close_all_on_nth:
                # 第 N 次触发：全平并重置
                high_since_entry[col] = np.nan
                close_count[col] = 0
                return nb.close_position_nb(price=close_now, fees=0.0003, slippage=0.0005)
            # 第 1、2 次：平仓 close_pct
            size_close = c.position_now * close_pct
            if size_close < min_position or c.position_now - size_close < min_position:
                high_since_entry[col] = np.nan
                close_count[col] = 0
                return nb.close_position_nb(price=close_now, fees=0.0003, slippage=0.0005)
            return nb.order_nb(
                size=-size_close,
                price=close_now,
                direction=Direction.LongOnly,
                fees=0.0003,
                slippage=0.0005,
            )

    return NoOrder


DATA_DIR = os.path.join(_REPO_ROOT, "stock_data")


def run_atr_trailing_demo(symbol: str = "0700.HK"):
    """加载数据、计算 ATR、生成简单进出场，用 ATR*2 移动止损回测。"""
    data = load_data(DATA_DIR, symbols=[symbol])
    if not data or symbol not in data:
        print(f"无数据: {symbol}")
        return

    df = data[symbol].tail(365)
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # 单标的为 Series，统一成 DataFrame 一列以便与 2D 数组对齐
    if isinstance(close, pd.Series):
        open_ = open_.to_frame(symbol)
        high = high.to_frame(symbol)
        low = low.to_frame(symbol)
        close = close.to_frame(symbol)

    atr_df = calc_atr(high, low, close, period=ATR_PERIOD)
    atr_arr = np.asarray(atr_df)
    high_arr = np.asarray(high)

    # 简单信号示例：收盘上穿 20 日均线入场，否则无出场信号（由 ATR 止损/后续可加其他出场）
    ma = close.rolling(20).mean()
    entries = (close > ma) & (close.shift(1) <= ma.shift(1))
    exits = pd.DataFrame(False, index=close.index, columns=close.columns)

    # 必须传入 open/high/low 才能使用 stop 相关逻辑
    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        open=open_,
        high=high,
        low=low,
        adjust_sl_func_nb=adjust_sl_atr_nb,
        adjust_sl_args=(atr_arr, high_arr, ATR_MULT),
        use_stops=True,
        init_cash=1_000_000,
        fees=0.0003,
        slippage=0.0005,
        freq="1D",
    )

    print(f"标的: {symbol} | ATR 周期: {ATR_PERIOD} | 止损: 回撤 {ATR_MULT}*ATR")
    tr = pf.total_return()
    md = pf.max_drawdown()
    print(f"总收益率: {tr.iloc[-1] if hasattr(tr, 'iloc') else float(tr):.4f}")
    print(f"最大回撤: {md.iloc[-1] if hasattr(md, 'iloc') else float(md):.4f}")
    print(f"交易次数: {int(pf.trades.count().sum())}")
    return pf


def run_atr_strict_peak_demo(symbol: str = "0700.HK"):
    """严格 trailing：自入场以来最高价 peak，当 low <= peak − ATR×mult 时止损。使用 from_order_func 维护 peak。"""
    data = load_data(DATA_DIR, symbols=[symbol])
    if not data or symbol not in data:
        print(f"无数据: {symbol}")
        return

    df = data[symbol].tail(365)
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]

    if isinstance(close, pd.Series):
        open_ = open_.to_frame(symbol)
        high = high.to_frame(symbol)
        low = low.to_frame(symbol)
        close = close.to_frame(symbol)

    atr_df = calc_atr(high, low, close, period=ATR_PERIOD)
    n_rows, n_cols = close.shape[0], close.shape[1]

    ma = close.rolling(20).mean()
    entries = (close > ma) & (close.shift(1) <= ma.shift(1))
    exits = pd.DataFrame(False, index=close.index, columns=close.columns)

    high_since_entry = np.full(n_cols, np.nan, dtype=np.float64)
    close_arr = np.asarray(close)
    high_arr = np.asarray(high)
    low_arr = np.asarray(low)
    atr_arr = np.asarray(atr_df)
    entries_arr = np.asarray(entries, dtype=np.bool_)
    exits_arr = np.asarray(exits, dtype=np.bool_)

    pf = vbt.Portfolio.from_order_func(
        close,
        order_func_peak_atr_nb,
        high_since_entry,
        entries_arr,
        exits_arr,
        high_arr,
        low_arr,
        atr_arr,
        close_arr,
        ATR_MULT,
        init_cash=1_000_000,
        freq="1D",
    )

    print(f"[严格 trailing] 标的: {symbol} | 止损: peak − {ATR_MULT}×ATR（peak=入场以来最高价）")
    tr = pf.total_return()
    md = pf.max_drawdown()
    print(f"总收益率: {tr.iloc[-1] if hasattr(tr, 'iloc') else float(tr):.4f}")
    print(f"最大回撤: {md.iloc[-1] if hasattr(md, 'iloc') else float(md):.4f}")
    print(f"交易次数: {int(pf.trades.count().sum())}")
    return pf


def run_atr_combined_demo(symbol: str = "0700.HK"):
    """组合止损（满足其一即触发）：① 移动止损超过 20%（自 peak 回撤≥20%）；② peak − 2×ATR。"""
    data = load_data(DATA_DIR, symbols=[symbol])
    if not data or symbol not in data:
        print(f"无数据: {symbol}")
        return

    df = data[symbol].tail(365)
    high = df["high"]
    low = df["low"]
    close = df["close"]

    if isinstance(close, pd.Series):
        high = high.to_frame(symbol)
        low = low.to_frame(symbol)
        close = close.to_frame(symbol)

    atr_df = calc_atr(high, low, close, period=ATR_PERIOD)
    n_cols = close.shape[1]

    ma = close.rolling(20).mean()
    entries = (close > ma) & (close.shift(1) <= ma.shift(1))
    exits = pd.DataFrame(False, index=close.index, columns=close.columns)

    high_since_entry = np.full(n_cols, np.nan, dtype=np.float64)
    close_count = np.zeros(n_cols, dtype=np.int64)
    close_arr = np.asarray(close)
    high_arr = np.asarray(high)
    low_arr = np.asarray(low)
    atr_arr = np.asarray(atr_df)
    entries_arr = np.asarray(entries, dtype=np.bool_)
    exits_arr = np.asarray(exits, dtype=np.bool_)

    pf = vbt.Portfolio.from_order_func(
        close,
        order_func_combined_nb,
        high_since_entry,
        close_count,
        entries_arr,
        exits_arr,
        high_arr,
        low_arr,
        atr_arr,
        close_arr,
        SL_PCT,
        ATR_MULT,
        CLOSE_PCT,
        CLOSE_ALL_ON_NTH,
        MIN_POSITION,
        init_cash=1_000_000,
        freq="1D",
    )

    print(f"[组合止损] 标的: {symbol} | ① 回撤≥{SL_PCT*100:.0f}% 或 ② peak−{ATR_MULT}×ATR；第1、2次平仓 {CLOSE_PCT*100:.0f}%，第{CLOSE_ALL_ON_NTH}次清完")
    tr = pf.total_return()
    md = pf.max_drawdown()
    print(f"总收益率: {tr.iloc[-1] if hasattr(tr, 'iloc') else float(tr):.4f}")
    print(f"最大回撤: {md.iloc[-1] if hasattr(md, 'iloc') else float(md):.4f}")
    print(f"交易次数: {int(pf.trades.count().sum())}")
    return pf


def run_compare(symbol: str = "0700.HK"):
    """跑三种策略并输出对比表。"""
    results = []
    # 默认：from_signals + adjust_sl
    pf = run_atr_trailing_demo(symbol)
    if pf is not None:
        tr = pf.total_return()
        md = pf.max_drawdown()
        n = int(pf.trades.count().sum())
        results.append(("默认 (每 bar high−2×ATR)", tr.iloc[-1] if hasattr(tr, "iloc") else float(tr), md.iloc[-1] if hasattr(md, "iloc") else float(md), n))
    # strict
    pf = run_atr_strict_peak_demo(symbol)
    if pf is not None:
        tr = pf.total_return()
        md = pf.max_drawdown()
        n = int(pf.trades.count().sum())
        results.append(("严格 (peak−2×ATR 一次全平)", tr.iloc[-1] if hasattr(tr, "iloc") else float(tr), md.iloc[-1] if hasattr(md, "iloc") else float(md), n))
    # combined
    pf = run_atr_combined_demo(symbol)
    if pf is not None:
        tr = pf.total_return()
        md = pf.max_drawdown()
        n = int(pf.trades.count().sum())
        results.append(("组合 (20% 或 2×ATR，30%×2+第3次清完)", tr.iloc[-1] if hasattr(tr, "iloc") else float(tr), md.iloc[-1] if hasattr(md, "iloc") else float(md), n))
    print("\n" + "=" * 70)
    print(f"策略对比 | 标的: {symbol} | 近 365 日")
    print("=" * 70)
    print(f"{'策略':<45} {'总收益':>8} {'最大回撤':>10} {'交易次数':>8}")
    print("-" * 70)
    for name, ret, drawdown, count in results:
        print(f"{name:<45} {ret:>8.2%} {drawdown:>10.2%} {count:>8}")
    print("=" * 70)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a not in ("--strict", "--combined", "--compare")]
    strict = "--strict" in sys.argv
    combined = "--combined" in sys.argv
    compare = "--compare" in sys.argv
    symbol = args[0] if args else "0700.HK"
    if compare:
        run_compare(symbol)
    elif combined:
        run_atr_combined_demo(symbol)
    elif strict:
        run_atr_strict_peak_demo(symbol)
    else:
        run_atr_trailing_demo(symbol)
