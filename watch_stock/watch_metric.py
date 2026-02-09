"""
监控指标筛选：从 stock_list 读取股票，加载近一年数据，
筛选「最近 3 日内有日线金叉 或 最近 3 周内有周线金叉」任一即可的标的，
并展示：金叉/死叉(日线、周线)、周线形态(talib)、日线形态(talib)、成交量、KDJ(日)、KDJ(周)。
复用 backtesting 的 data_loader、indicators，不重复造轮子。
"""
import json
import os
import sys

import pandas as pd
import numpy as np

# 项目根目录（backtesting 上一级），便于导入 backtesting 包
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backtesting.data_loader import load_data
from backtesting.indicators import (
    calc_kdj,
    get_kdj_cross_signals,
    calc_weekly_kdj,
    resample_to_weekly,
)

DATA_DIR = os.path.join(_REPO_ROOT, "stock_data")
STOCK_LIST_PATH = os.path.join(DATA_DIR, "stock_list.json")

# 最近一年：按自然日取约 365 天（实际会按数据量取）
LOOKBACK_DAYS = 365
# 筛选：最近 3 日内有日线金叉 或 最近 3 周内有周线金叉 任一即可
LAST_N_DAYS = 3
LAST_N_WEEKS = 3
# KDJ 金叉/死叉 J 阈值
KDJ_J_THRESHOLD_LONG = 35   # 金叉：J 超卖区
KDJ_J_THRESHOLD_SHORT = 80  # 死叉：J 超买区

try:
    import talib
except ImportError:
    talib = None


def load_stock_list(path: str) -> list:
    """从 stock_list.json 读取股票代码列表。"""
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def get_talib_pattern_names_at_bar(open_series, high_series, low_series, close_series, bar_index=-1):
    """
    在指定 K 线位置检测 TA-Lib 形态，返回该 bar 触发的看多形态名称列表。
    bar_index: -1 表示最后一根。
    """
    if talib is None:
        return []
    try:
        groups = talib.get_function_groups()
        pattern_names = groups.get("Pattern Recognition", [])
    except Exception:
        return []
    o = open_series.values.astype(float)
    h = high_series.values.astype(float)
    l = low_series.values.astype(float)
    c = close_series.values.astype(float)
    idx = bar_index if bar_index >= 0 else len(c) + bar_index
    if idx < 0 or idx >= len(c):
        return []
    names = []
    for name in pattern_names:
        if not hasattr(talib, name):
            continue
        try:
            func = getattr(talib, name)
            res = func(o, h, l, c)
            if res is not None and len(res) > idx and res[idx] == 100:
                names.append(name)
        except Exception:
            continue
    return names


def run():
    # 1. 读取股票列表
    symbols = load_stock_list(STOCK_LIST_PATH)
    if not symbols:
        print("stock_list.json 为空或不存在。")
        return

    # 2. 加载日线数据（仅加载列表中的股票）
    data_map = load_data(DATA_DIR, symbols=symbols, period_suffix="day")
    if not data_map:
        print("未加载到任何数据，请检查 DATA_DIR 下是否有对应 CSV。")
        return

    # 3. 取最近一年
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=LOOKBACK_DAYS)
    data_map = {
        sym: df[df.index >= cutoff].copy()
        for sym, df in data_map.items()
        if len(df[df.index >= cutoff]) >= 50  # 至少约 2 个月数据
    }
    if not data_map:
        print("截取最近一年后无足够数据。")
        return

    print(f"共 {len(data_map)} 只股票参与筛选（最近约一年数据）。")

    # 4. 筛选：最近 3 日内有日线金叉 或 最近 3 周内有周线金叉 任一即可
    results = []
    for sym, df in data_map.items():
        open_s = df["open"]
        high_s = df["high"]
        low_s = df["low"]
        close_s = df["close"]
        vol_s = df["volume"] if "volume" in df.columns else pd.Series(0.0, index=df.index)

        open_df = pd.DataFrame({sym: open_s})
        high_df = pd.DataFrame({sym: high_s})
        low_df = pd.DataFrame({sym: low_s})
        close_df = pd.DataFrame({sym: close_s})

        # 日线 KDJ、金叉、死叉
        k, d, j = calc_kdj(close_s, high_s, low_s)
        daily_gold = get_kdj_cross_signals(k, d, j, threshold=KDJ_J_THRESHOLD_LONG, mode="long")
        daily_death = get_kdj_cross_signals(k, d, j, threshold=KDJ_J_THRESHOLD_SHORT, mode="short")
        last_3_days_gold = daily_gold.tail(LAST_N_DAYS).any()
        last_3_days_death = daily_death.tail(LAST_N_DAYS).any()

        # 周线 KDJ 在周线维度上计算
        w_open, w_high, w_low, w_close = resample_to_weekly(open_df, high_df, low_df, close_df)
        wc, wh, wl = w_close[sym], w_high[sym], w_low[sym]
        wk_series, wd_series, wj_series = calc_kdj(wc, wh, wl, N=9, M=3)
        weekly_gold = get_kdj_cross_signals(wk_series, wd_series, wj_series, threshold=KDJ_J_THRESHOLD_LONG, mode="long")
        weekly_death = get_kdj_cross_signals(wk_series, wd_series, wj_series, threshold=KDJ_J_THRESHOLD_SHORT, mode="short")
        if len(weekly_gold) < LAST_N_WEEKS:
            continue
        last_3_weeks_gold = weekly_gold.tail(LAST_N_WEEKS).any()
        last_3_weeks_death = weekly_death.tail(LAST_N_WEEKS).any()

        # 筛选：最近 3 日有日线金叉 或 最近 3 周有周线金叉 任一即可
        if not (last_3_days_gold or last_3_weeks_gold):
            continue

        # 周线 KDJ 对齐到日线索引（用于展示最后一日的 KDJ 周）
        dk, dd, dj = calc_weekly_kdj(open_df, high_df, low_df, close_df, N=9, M=3)
        wk, wd, wj = dk[sym], dd[sym], dj[sym]

        # 取最近一根的展示数据
        last = df.index[-1]
        k_daily = k.iloc[-1] if not pd.isna(k.iloc[-1]) else None
        d_daily = d.iloc[-1] if not pd.isna(d.iloc[-1]) else None
        j_daily = j.iloc[-1] if not pd.isna(j.iloc[-1]) else None
        k_weekly = wk.iloc[-1] if not pd.isna(wk.iloc[-1]) else None
        d_weekly = wd.iloc[-1] if not pd.isna(wd.iloc[-1]) else None
        j_weekly = wj.iloc[-1] if not pd.isna(wj.iloc[-1]) else None
        volume_last = vol_s.iloc[-1] if len(vol_s) else None

        # 日线形态：最后一根 K 线
        daily_patterns = get_talib_pattern_names_at_bar(open_s, high_s, low_s, close_s, -1)
        daily_pattern_str = ",".join(daily_patterns) if daily_patterns else "-"

        # 周线形态：先 resample 成周线，再取最后一根
        w_open, w_high, w_low, w_close = resample_to_weekly(
            open_df, high_df, low_df, close_df
        )
        if len(w_close) == 0:
            weekly_pattern_str = "-"
        else:
            wo = w_open[sym]
            wh = w_high[sym]
            wl = w_low[sym]
            wc = w_close[sym]
            weekly_patterns = get_talib_pattern_names_at_bar(wo, wh, wl, wc, -1)
            weekly_pattern_str = ",".join(weekly_patterns) if weekly_patterns else "-"

        results.append({
            "股票": sym,
            "日线金叉(近3日)": "是" if last_3_days_gold else "否",
            "日线死叉(近3日)": "是" if last_3_days_death else "否",
            "周线金叉(近3周)": "是" if last_3_weeks_gold else "否",
            "周线死叉(近3周)": "是" if last_3_weeks_death else "否",
            "周线形态(talib)": weekly_pattern_str,
            "日线形态(talib)": daily_pattern_str,
            "成交量": volume_last,
            "KDJ(日)K": round(k_daily, 2) if k_daily is not None else "-",
            "KDJ(日)D": round(d_daily, 2) if d_daily is not None else "-",
            "KDJ(日)J": round(j_daily, 2) if j_daily is not None else "-",
            "KDJ(周)K": round(k_weekly, 2) if k_weekly is not None else "-",
            "KDJ(周)D": round(d_weekly, 2) if d_weekly is not None else "-",
            "KDJ(周)J": round(j_weekly, 2) if j_weekly is not None else "-",
        })

    # 5. 展示（Markdown 表格）
    if not results:
        print("无符合「最近 3 日内有日线金叉 或 最近 3 周内有周线金叉」的股票。")
        return

    out = pd.DataFrame(results)
    print("\n符合条件的股票：\n")
    print(_df_to_markdown_table(out))
    return out


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """将 DataFrame 转为 Markdown 表格字符串。"""
    cols = list(df.columns)
    lines = []
    # 表头
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    # 数据行
    for _, row in df.iterrows():
        cells = [str(row[c]) if pd.notna(row[c]) else "" for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    run()
