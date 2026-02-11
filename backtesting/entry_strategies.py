"""
统一入场策略模块：各策略返回与 close 同索引的 bool Series/DataFrame，便于用 & | 组合。

用法示例：
    from backtesting.entry_strategies import (
        entry_ma, entry_123, entry_2b, entry_kdj_week,
        exit_123, exit_2b,
    )

    # 单策略
    entries = entry_ma(close, period=20)

    # 组合：同时满足 2B 与 123 才入场
    entries = entry_2b(high, low, close) & entry_123(high, low, close)

    # 组合：123 或 2B 任一即入场
    entries = entry_123(high, low, close) | entry_2b(high, low, close)

    # 组合：周 KDJ 金叉 且 均线金叉
    entries = entry_kdj_week(close, high, low) & entry_ma(close, period=20)

    # 出场：123 或 2B 任一触发
    exits = exit_123(high, low, close) | exit_2b(high, low, close)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union

from backtesting.indicators import calc_kdj, get_kdj_cross_signals, resample_to_weekly

# ---------- 道氏 123/2B 用到的拐点与趋势线（内部） ----------
DEFAULT_PIVOT_ORDER = 5
DEFAULT_LOOKBACK = 135


def _pivot_lows(low: np.ndarray, order: int) -> np.ndarray:
    """标记 pivot low：low[i] 为 [i-order, i] 内最小值则置 1。"""
    n = len(low)
    out = np.zeros(n, dtype=np.float64)
    for i in range(order, n):
        if low[i] <= low[i - order : i + 1].min():
            out[i] = 1.0
    return out


def _pivot_highs(high: np.ndarray, order: int) -> np.ndarray:
    """标记 pivot high：high[i] 为 [i-order, i] 内最大值则置 1。"""
    n = len(high)
    out = np.zeros(n, dtype=np.float64)
    for i in range(order, n):
        if high[i] >= high[i - order : i + 1].max():
            out[i] = 1.0
    return out


def _last_two_pivot_indices(pivot_mask: np.ndarray, from_end: int) -> tuple:
    """在 [0, from_end] 内找最后两个 pivot 的索引，返回 (idx_second, idx_first)。"""
    idx_first = idx_second = None
    for i in range(from_end, -1, -1):
        if pivot_mask[i] != 0:
            if idx_first is None:
                idx_first = i
            elif idx_second is None:
                idx_second = i
                break
    return idx_second, idx_first


def _trend_line_value_at(
    bar_idx: int, idx1: int, val1: float, idx2: int, val2: float
) -> float:
    """两点 (idx1, val1), (idx2, val2) 连线在 bar_idx 处的值。"""
    if idx1 is None or idx2 is None or idx1 == idx2:
        return np.nan
    slope = (val2 - val1) / (idx2 - idx1)
    return val1 + slope * (bar_idx - idx1)


def _build_entry_123_2b_single(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    order: int,
) -> tuple:
    """
    单列 OHLC 的 123/2B 入场信号（仅入场）。
    返回 (entry_123, entry_2b)，均为 bool 数组。
    """
    n = len(close)
    pl = _pivot_lows(low, order)
    ph = _pivot_highs(high, order)
    entry_123 = np.zeros(n, dtype=np.bool_)
    entry_2b = np.zeros(n, dtype=np.bool_)

    for i in range(2 * order, n):
        ph_idx2, ph_idx1 = _last_two_pivot_indices(ph, i - 1)
        down_trend_line = np.nan
        if ph_idx1 is not None:
            down_trend_line = (
                high[ph_idx1]
                if ph_idx2 is None
                else _trend_line_value_at(i, ph_idx2, high[ph_idx2], ph_idx1, high[ph_idx1])
            )

        _, last_ph_idx = _last_two_pivot_indices(ph, i - 1)
        _, last_pl_idx = _last_two_pivot_indices(pl, i - 1)
        prev_pl_idx, _ = (
            _last_two_pivot_indices(pl, last_pl_idx - 1)
            if last_pl_idx is not None and last_pl_idx > 0
            else (None, None)
        )

        # 入场 123：下降趋势反转
        no_new_low = False
        if last_pl_idx is not None and prev_pl_idx is not None:
            no_new_low = low[last_pl_idx] > low[prev_pl_idx]
        prev_swing_high = np.nan
        if last_pl_idx is not None and last_pl_idx > 0:
            _, prev_ph_before_pl = _last_two_pivot_indices(ph, last_pl_idx - 1)
            if prev_ph_before_pl is not None:
                prev_swing_high = high[prev_ph_before_pl]
        if (
            not np.isnan(down_trend_line)
            and close[i] > down_trend_line
            and no_new_low
            and not np.isnan(prev_swing_high)
            and close[i] > prev_swing_high
        ):
            entry_123[i] = True

        # 入场 2B：假跌破
        if last_pl_idx is not None:
            prior_low = low[last_pl_idx]
            broke_below = np.any(low[last_pl_idx + 1 : i + 1] < prior_low)
            if broke_below and close[i] > prior_low:
                entry_2b[i] = True

    return entry_123, entry_2b


def _build_exit_123_2b_single(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    order: int,
) -> tuple:
    """
    单列 OHLC 的 123/2B 出场信号。
    返回 (exit_123, exit_2b)，均为 bool 数组。
    """
    n = len(close)
    pl = _pivot_lows(low, order)
    ph = _pivot_highs(high, order)
    exit_123 = np.zeros(n, dtype=np.bool_)
    exit_2b = np.zeros(n, dtype=np.bool_)

    for i in range(2 * order, n):
        pl_idx2, pl_idx1 = _last_two_pivot_indices(pl, i - 1)
        up_trend_line = np.nan
        if pl_idx1 is not None:
            up_trend_line = (
                low[pl_idx1]
                if pl_idx2 is None
                else _trend_line_value_at(i, pl_idx2, low[pl_idx2], pl_idx1, low[pl_idx1])
            )

        _, last_ph_idx = _last_two_pivot_indices(ph, i - 1)
        prev_ph_idx, _ = (
            _last_two_pivot_indices(ph, last_ph_idx - 1)
            if last_ph_idx is not None and last_ph_idx > 0
            else (None, None)
        )

        # 出场 123：上升趋势反转
        no_new_high = False
        if last_ph_idx is not None and prev_ph_idx is not None:
            no_new_high = high[last_ph_idx] < high[prev_ph_idx]
        prev_swing_low = np.nan
        if last_ph_idx is not None and last_ph_idx > 0:
            _, prev_pl_before_ph = _last_two_pivot_indices(pl, last_ph_idx - 1)
            if prev_pl_before_ph is not None:
                prev_swing_low = low[prev_pl_before_ph]
        if (
            not np.isnan(up_trend_line)
            and close[i] < up_trend_line
            and no_new_high
            and not np.isnan(prev_swing_low)
            and close[i] < prev_swing_low
        ):
            exit_123[i] = True

        # 出场 2B：假突破前高后跌破
        if last_ph_idx is not None:
            prior_high = high[last_ph_idx]
            broke_above = np.any(high[last_ph_idx + 1 : i + 1] > prior_high)
            if broke_above and close[i] < prior_high:
                exit_2b[i] = True

    return exit_123, exit_2b


# ---------- 对外接口：统一接受 Series/DataFrame，返回同形状 bool ----------

EntryType = Union[pd.Series, pd.DataFrame]


def entry_ma(
    close: EntryType,
    period: int = 20,
) -> EntryType:
    """
    均线金叉：收盘上穿 period 日均线。
    """
    ma = close.rolling(period).mean()
    return (close > ma) & (close.shift(1) <= ma.shift(1))


def entry_123(
    high: EntryType,
    low: EntryType,
    close: EntryType,
    pivot_order: int = DEFAULT_PIVOT_ORDER,
    lookback: int = DEFAULT_LOOKBACK,
) -> EntryType:
    """
    道氏 123 入场：下降趋势反转 = ① 突破下降趋势线 ② 不再创新低 ③ 突破前反弹高点。
    """
    if isinstance(close, pd.Series):
        e123, _ = _build_entry_123_2b_single(
            high.values, low.values, close.values, pivot_order
        )
        return pd.Series(e123, index=close.index, name=close.name)
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=bool)
    for col in close.columns:
        e123, _ = _build_entry_123_2b_single(
            high[col].values, low[col].values, close[col].values, pivot_order
        )
        out[col] = e123
    return out


def entry_2b(
    high: EntryType,
    low: EntryType,
    close: EntryType,
    pivot_order: int = DEFAULT_PIVOT_ORDER,
    lookback: int = DEFAULT_LOOKBACK,
) -> EntryType:
    """
    道氏 2B 入场：假跌破前低后收盘站回前低之上。
    """
    if isinstance(close, pd.Series):
        _, e2b = _build_entry_123_2b_single(
            high.values, low.values, close.values, pivot_order
        )
        return pd.Series(e2b, index=close.index, name=close.name)
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=bool)
    for col in close.columns:
        _, e2b = _build_entry_123_2b_single(
            high[col].values, low[col].values, close[col].values, pivot_order
        )
        out[col] = e2b
    return out


def exit_123(
    high: EntryType,
    low: EntryType,
    close: EntryType,
    pivot_order: int = DEFAULT_PIVOT_ORDER,
    lookback: int = DEFAULT_LOOKBACK,
) -> EntryType:
    """
    道氏 123 出场：上升趋势反转 = ① 跌破上升趋势线 ② 不再创新高 ③ 跌破前回档低点。
    """
    if isinstance(close, pd.Series):
        ex, _ = _build_exit_123_2b_single(
            high.values, low.values, close.values, pivot_order
        )
        return pd.Series(ex, index=close.index, name=close.name)
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=bool)
    for col in close.columns:
        ex, _ = _build_exit_123_2b_single(
            high[col].values, low[col].values, close[col].values, pivot_order
        )
        out[col] = ex
    return out


def exit_2b(
    high: EntryType,
    low: EntryType,
    close: EntryType,
    pivot_order: int = DEFAULT_PIVOT_ORDER,
    lookback: int = DEFAULT_LOOKBACK,
) -> EntryType:
    """
    道氏 2B 出场：假突破前高后收盘跌破前高。
    """
    if isinstance(close, pd.Series):
        _, ex = _build_exit_123_2b_single(
            high.values, low.values, close.values, pivot_order
        )
        return pd.Series(ex, index=close.index, name=close.name)
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=bool)
    for col in close.columns:
        _, ex = _build_exit_123_2b_single(
            high[col].values, low[col].values, close[col].values, pivot_order
        )
        out[col] = ex
    return out


def entry_kdj_week(
    close: EntryType,
    high: EntryType,
    low: EntryType,
    n: int = 9,
    m: int = 3,
    j_buy: float = 30,
    open_: EntryType | None = None,
) -> EntryType:
    """
    周 KDJ 金叉且 J <= j_buy 作为入场；周线对齐到日线（ffill），当日对应周满足则入场。
    若未传 open_，周线重采样时 open 用 close.shift(1) 近似。
    """
    if open_ is None:
        open_ = close.shift(1).fillna(close)
    if isinstance(close, pd.Series):
        o = open_.to_frame("x") if isinstance(open_, pd.Series) else open_
        h = high.to_frame("x") if isinstance(high, pd.Series) else high
        l = low.to_frame("x") if isinstance(low, pd.Series) else low
        c = close.to_frame("x") if isinstance(close, pd.Series) else close
        w_open, w_high, w_low, w_close = resample_to_weekly(o, h, l, c)
        wk, wd, wj = calc_kdj(w_close["x"], w_high["x"], w_low["x"], N=n, M=m)
        cross = get_kdj_cross_signals(wk, wd, wj, threshold=j_buy, mode="long")
        daily = cross.reindex(close.index, method="ffill").fillna(False)
        return daily
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=bool)
    for col in close.columns:
        o = open_[col] if isinstance(open_, pd.DataFrame) else open_
        h = high[col] if isinstance(high, pd.DataFrame) else high
        l = low[col] if isinstance(low, pd.DataFrame) else low
        c = close[col] if isinstance(close, pd.DataFrame) else close
        out[col] = entry_kdj_week(c, h, l, n=n, m=m, j_buy=j_buy, open_=o)
    return out


__all__ = [
    "entry_ma",
    "entry_123",
    "entry_2b",
    "entry_kdj_week",
    "exit_123",
    "exit_2b",
]
