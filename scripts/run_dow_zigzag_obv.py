"""
道氏理论验证：ZigZag(5%) 波峰波谷 + 123 法则 + OBV 量能确认。

1. ZigZag 深度约 5%：自动勾勒波峰波谷，屏蔽 K 线级噪音。
2. 日线级别：屏蔽短线投机噪音。
3. 123 法则（基于 ZigZag）：
   - 趋势线被突破
   - 上升趋势不再创新高（或下降趋势不再创新低）
   - 价格跌破前一个波谷（或升破前一个波峰）
4. OBV 辅助：价创新高但 OBV 未创新高 → 视为假突破/噪音，可过滤。

运行（项目根目录）：
  .venv/bin/python scripts/run_dow_zigzag_obv.py [symbol] [--no-obv] [--entry ma|123] [--no-chart]
  不传 symbol 默认 0700.HK。
  --no-obv：关闭 OBV 背离过滤。
  --entry ma：入场用均线金叉。
  --no-chart：不输出图表 HTML。
"""
import os
import sys
import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backtesting.data_loader import load_data
from backtesting.indicators import calc_obv, zigzag_pivots, zigzag_pivots_mt4
import vectorbt as vbt

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    make_subplots = None

# ZigZag：使用 MT4 方案（Depth=12, Deviation=5%, Backstep=3）
ZZ_MT4_DEPTH = 12
ZZ_MT4_DEVIATION = 0.05
ZZ_MT4_BACKSTEP = 3
# 兼容旧版纯比例 ZigZag（仅当未用 MT4 时）
ZZ_DEPTH_PCT = 0.05
# 入场：均线金叉（日线）
MA_ENTRY = 20
# 是否用 OBV 背离过滤 123 信号（默认 True）
USE_OBV_FILTER = True

DATA_DIR = os.path.join(_REPO_ROOT, "stock_data")
REPORT_DIR = os.path.join(_REPO_ROOT, "reports")


def _trend_line_value_at(bar_idx: int, idx1: int, val1: float, idx2: int, val2: float) -> float:
    if idx1 is None or idx2 is None or idx1 == idx2:
        return np.nan
    slope = (val2 - val1) / (idx2 - idx1)
    return val1 + slope * (bar_idx - idx1)


def _last_two_pivots(pivots: list, is_high: bool) -> tuple:
    """取 pivots 中最后两个波峰或波谷的 (idx, val)。返回 ((idx2, val2), (idx1, val1)) 或 (None, (idx1, val1))。"""
    sub = [p for p in pivots if p[2] == is_high]
    if not sub:
        return (None, None), (None, None)
    if len(sub) == 1:
        return (None, None), (sub[-1][0], sub[-1][1])
    return (sub[-2][0], sub[-2][1]), (sub[-1][0], sub[-1][1])


def _plot_price_zigzag_signals(
    index: pd.DatetimeIndex,
    close: np.ndarray,
    pivots: list,
    entry_mask: np.ndarray,
    exit_mask: np.ndarray,
    symbol: str,
    title_suffix: str = "",
    obv: np.ndarray | None = None,
) -> "go.Figure":
    """绘制：收盘价 + ZigZag 折线 + 趋势线 + 支撑/压力 + 段起点/终点/回撤点 + 入场/出场；可选 OBV。"""
    if go is None or make_subplots is None:
        return None

    has_obv = obv is not None and len(obv) == len(index)
    if has_obv:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.55, 0.45],
            subplot_titles=("价格 + ZigZag + 趋势线 + 支撑/压力", "OBV 能量潮"),
        )
        row_price, row_obv = 1, 2
    else:
        fig = go.Figure()

    n = len(index)

    def add_price_traces(fig_obj, row=None):
        kw = dict() if row is None else dict(row=row, col=1)
        fig_obj.add_trace(
            go.Scatter(
                x=index,
                y=close,
                name="收盘价",
                line=dict(color="rgb(80,120,180)", width=1.5),
            ),
            **kw,
        )
        if pivots:
            x_zz = [index[p[0]] for p in pivots]
            y_zz = [p[1] for p in pivots]
            fig_obj.add_trace(
                go.Scatter(
                    x=x_zz,
                    y=y_zz,
                    mode="lines+markers",
                    name="ZigZag",
                    line=dict(color="rgba(220,120,0,0.95)", width=2.5),
                    marker=dict(size=6, color="rgba(220,120,0,0.95)", line=dict(width=1, color="darkorange")),
                ),
                **kw,
            )
            # 方法二：趋势线（最近两低点连线、最近两高点连线，延长到图表右侧）
            (pl2, pl1) = _last_two_pivots(pivots, is_high=False)
            (ph2, ph1) = _last_two_pivots(pivots, is_high=True)
            if pl1[0] is not None and pl2[0] is not None:
                end_val = _trend_line_value_at(n - 1, pl2[0], pl2[1], pl1[0], pl1[1])
                fig_obj.add_trace(
                    go.Scatter(
                        x=[index[pl2[0]], index[pl1[0]], index[-1]],
                        y=[pl2[1], pl1[1], end_val],
                        mode="lines",
                        name="上升趋势线",
                        line=dict(color="rgba(80,180,80,0.7)", width=1.5, dash="dash"),
                    ),
                    **kw,
                )
            if ph1[0] is not None and ph2[0] is not None:
                end_val = _trend_line_value_at(n - 1, ph2[0], ph2[1], ph1[0], ph1[1])
                fig_obj.add_trace(
                    go.Scatter(
                        x=[index[ph2[0]], index[ph1[0]], index[-1]],
                        y=[ph2[1], ph1[1], end_val],
                        mode="lines",
                        name="下降趋势线",
                        line=dict(color="rgba(200,80,80,0.7)", width=1.5, dash="dash"),
                    ),
                    **kw,
                )
            # 方法三：支撑/压力（最近 2 个低点画支撑，最近 2 个高点画压力，水平线段）
            last_lows = [p for p in pivots if not p[2]][-2:]
            last_highs = [p for p in pivots if p[2]][-2:]
            x_end = max(1, n // 15)
            for i, (idx, val, _) in enumerate(last_lows):
                end_idx = min(idx + x_end, n - 1)
                fig_obj.add_trace(
                    go.Scatter(
                        x=[index[idx], index[end_idx]],
                        y=[val, val],
                        mode="lines",
                        name="支撑",
                        line=dict(color="rgba(80,160,80,0.5)", width=1, dash="dot"),
                        showlegend=(i == 0),
                    ),
                    **kw,
                )
            for i, (idx, val, _) in enumerate(last_highs):
                end_idx = min(idx + x_end, n - 1)
                fig_obj.add_trace(
                    go.Scatter(
                        x=[index[idx], index[end_idx]],
                        y=[val, val],
                        mode="lines",
                        name="压力",
                        line=dict(color="rgba(200,80,80,0.5)", width=1, dash="dot"),
                        showlegend=(i == 0),
                    ),
                    **kw,
                )
            # 方法一：段起点 / 段终点 / 回撤点 标注（最后 3 个拐点）
            labels = ["段起点", "段终点", "回撤点"]
            for i, p in enumerate(pivots[-3:]):
                lab = labels[i]
                fig_obj.add_trace(
                    go.Scatter(
                        x=[index[p[0]]],
                        y=[p[1]],
                        mode="markers+text",
                        name=lab,
                        text=[lab],
                        textposition="top center",
                        textfont=dict(size=10),
                        marker=dict(size=10, symbol="diamond", color="purple", line=dict(width=1)),
                        showlegend=True,
                    ),
                    **kw,
                )
        ent_idx = np.where(entry_mask)[0]
        ex_idx = np.where(exit_mask)[0]
        if len(ent_idx) > 0:
            fig_obj.add_trace(
                go.Scatter(
                    x=index[ent_idx],
                    y=close[ent_idx],
                    mode="markers",
                    name="入场",
                    marker=dict(symbol="triangle-up", size=12, color="green", line=dict(width=2, color="darkgreen")),
                ),
                **kw,
            )
        if len(ex_idx) > 0:
            fig_obj.add_trace(
                go.Scatter(
                    x=index[ex_idx],
                    y=close[ex_idx],
                    mode="markers",
                    name="出场",
                    marker=dict(symbol="triangle-down", size=12, color="red", line=dict(width=2, color="darkred")),
                ),
                **kw,
            )

    add_price_traces(fig, row_price if has_obv else None)

    if has_obv:
        fig.add_trace(
            go.Scatter(
                x=index,
                y=obv,
                name="OBV",
                line=dict(color="rgb(100,80,160)", width=1.5),
            ),
            row=row_obv,
            col=1,
        )
        fig.update_yaxes(title_text="价格", row=1, col=1)
        fig.update_yaxes(title_text="OBV", row=2, col=1)
        fig.update_xaxes(title_text="日期", row=2, col=1)
        fig.update_layout(
            title=f"{symbol} 日线 | 价格 + ZigZag + OBV {title_suffix}".strip(),
            template="plotly_white",
            height=700,
            width=1200,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            showlegend=True,
        )
    else:
        fig.update_layout(
            title=f"{symbol} 日线 | 价格 + ZigZag + 123 信号 {title_suffix}".strip(),
            xaxis_title="日期",
            yaxis_title="价格",
            template="plotly_white",
            height=500,
            width=1200,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
    return fig


def _last_two_zz(pivots: list, from_bar: int, is_high: bool) -> tuple:
    """从 pivots 中取 index <= from_bar 的最后两个波峰或波谷。(idx_second, idx_first), (val_second, val_first)"""
    indices = [p[0] for p in pivots if p[0] <= from_bar and p[2] == is_high]
    values = [p[1] for p in pivots if p[0] <= from_bar and p[2] == is_high]
    if len(indices) < 1:
        return (None, None), (np.nan, np.nan)
    if len(indices) < 2:
        return (None, indices[-1]), (np.nan, values[-1])
    return (indices[-2], indices[-1]), (values[-2], values[-1])


def build_123_zigzag_obv(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    depth_pct: float = ZZ_DEPTH_PCT,
    use_obv: bool = True,
    use_mt4_zigzag: bool = True,
    zz_depth: int = ZZ_MT4_DEPTH,
    zz_deviation: float = ZZ_MT4_DEVIATION,
    zz_backstep: int = ZZ_MT4_BACKSTEP,
) -> tuple:
    """
    基于 ZigZag 的 123 法则 + OBV 背离过滤。
    返回 (entry_123, exit_123)，均为 bool 数组，与 close 同长。
    use_mt4_zigzag=True 时使用 MT4 方案（depth, deviation, backstep），否则使用纯比例 depth_pct。
    """
    n = len(close)
    if use_mt4_zigzag:
        pivots = zigzag_pivots_mt4(high, low, close, depth=zz_depth, deviation=zz_deviation, backstep=zz_backstep)
    else:
        pivots = zigzag_pivots(high, low, close, depth_pct=depth_pct)
    obv = calc_obv(pd.Series(close), pd.Series(volume)).values if use_obv else None

    entry_123 = np.zeros(n, dtype=np.bool_)
    exit_123 = np.zeros(n, dtype=np.bool_)

    for i in range(1, n):
        # ----- 上升趋势线（最近两个 ZigZag 低点连线）
        (pl_idx2, pl_idx1), (pl_val2, pl_val1) = _last_two_zz(pivots, i - 1, is_high=False)
        if pl_idx1 is None:
            up_trend_line = np.nan
        elif pl_idx2 is None:
            up_trend_line = pl_val1
        else:
            up_trend_line = _trend_line_value_at(i, pl_idx2, pl_val2, pl_idx1, pl_val1)

        # ----- 下降趋势线（最近两个 ZigZag 高点连线）
        (ph_idx2, ph_idx1), (ph_val2, ph_val1) = _last_two_zz(pivots, i - 1, is_high=True)
        if ph_idx1 is None:
            down_trend_line = np.nan
        elif ph_idx2 is None:
            down_trend_line = ph_val1
        else:
            down_trend_line = _trend_line_value_at(i, ph_idx2, ph_val2, ph_idx1, ph_val1)

        # ----- 前一个波峰、前一个波谷（用于 123 的“跌破前波谷/升破前波峰”）
        (_, last_ph_idx), (_, last_ph_val) = _last_two_zz(pivots, i - 1, is_high=True)
        (_, prev_ph_idx), (_, prev_ph_val) = _last_two_zz(pivots, (last_ph_idx - 1) if last_ph_idx is not None and last_ph_idx > 0 else -1, is_high=True)
        (_, last_pl_idx), (_, last_pl_val) = _last_two_zz(pivots, i - 1, is_high=False)
        (_, prev_pl_idx), (_, prev_pl_val) = _last_two_zz(pivots, (last_pl_idx - 1) if last_pl_idx is not None and last_pl_idx > 0 else -1, is_high=False)

        # ---------- 出场 123：上升趋势反转
        no_new_high = last_ph_idx is not None and prev_ph_idx is not None and last_ph_val < prev_ph_val
        break_prev_low = prev_pl_val is not None and not np.isnan(prev_pl_val) and close[i] < prev_pl_val
        cond_exit = (
            not np.isnan(up_trend_line) and close[i] < up_trend_line
            and no_new_high
            and break_prev_low
        )
        if cond_exit and use_obv and obv is not None and last_ph_idx is not None and prev_ph_idx is not None:
            # OBV 顶背离：价创新高但 OBV 未创新高 → 确认反转
            obv_at_last_high = obv[last_ph_idx]
            obv_at_prev_high = obv[prev_ph_idx]
            if obv_at_last_high >= obv_at_prev_high:
                cond_exit = False  # 无背离，过滤掉
        if cond_exit:
            exit_123[i] = True

        # ---------- 入场 123：下降趋势反转
        no_new_low = last_pl_idx is not None and prev_pl_idx is not None and last_pl_val > prev_pl_val
        break_prev_high = prev_ph_val is not None and not np.isnan(prev_ph_val) and close[i] > prev_ph_val
        cond_entry = (
            not np.isnan(down_trend_line) and close[i] > down_trend_line
            and no_new_low
            and break_prev_high
        )
        if cond_entry and use_obv and obv is not None and last_pl_idx is not None and prev_pl_idx is not None:
            # OBV 底背离：价创新低但 OBV 未创新低 → 确认反转
            obv_at_last_low = obv[last_pl_idx]
            obv_at_prev_low = obv[prev_pl_idx]
            if obv_at_last_low <= obv_at_prev_low:
                cond_entry = False
        if cond_entry:
            entry_123[i] = True

    return entry_123, exit_123


def run_dow_zigzag_obv(
    symbol: str = "0700.HK",
    days: int = 365,
    use_obv_filter: bool = True,
    entry_mode: str = "123",
    save_charts: bool = True,
):
    """日线数据 + ZigZag(5%) + 123 出场 + 可选 OBV 背离过滤；入场可选 123 或均线金叉。"""
    data = load_data(DATA_DIR, symbols=[symbol])
    if not data or symbol not in data:
        print(f"无数据: {symbol}")
        return None

    df = data[symbol].tail(days)
    if "volume" not in df.columns or df["volume"].fillna(0).sum() == 0:
        print(f"无有效成交量，无法计算 OBV，将关闭 OBV 过滤")
        use_obv_filter = False
        df["volume"] = 1.0

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    volume = df["volume"].fillna(0).values
    index = df.index

    entry_123, exit_123 = build_123_zigzag_obv(
        high, low, close, volume,
        use_obv=use_obv_filter,
        use_mt4_zigzag=True,
        zz_depth=ZZ_MT4_DEPTH,
        zz_deviation=ZZ_MT4_DEVIATION,
        zz_backstep=ZZ_MT4_BACKSTEP,
    )

    close_df = df["close"].to_frame(symbol) if isinstance(df["close"], pd.Series) else df[["close"]].rename(columns={"close": symbol})
    if close_df.shape[1] != 1 and symbol not in close_df.columns:
        close_df = df["close"].to_frame(symbol)

    # 入场：123（纯道氏）或 均线金叉（更多交易以观察 123+OBV 出场）
    ma = close_df.rolling(MA_ENTRY).mean()
    entries_ma = (close_df > ma) & (close_df.shift(1) <= ma.shift(1))
    entries_123 = pd.DataFrame(entry_123, index=close_df.index, columns=close_df.columns)
    entries = entries_123 if entry_mode == "123" else entries_ma
    exits = pd.DataFrame(exit_123, index=close_df.index, columns=close_df.columns)

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

    obv_label = "OBV 背离过滤" if use_obv_filter else "未用 OBV"
    entry_label = "入场123" if entry_mode == "123" else f"入场MA{MA_ENTRY}"
    print(f"[道氏 ZigZag+123+OBV] 标的: {symbol} | 日线 | ZigZag MT4(D={ZZ_MT4_DEPTH},Dev={ZZ_MT4_DEVIATION*100:.0f}%,B={ZZ_MT4_BACKSTEP}) | {obv_label} | {entry_label}")
    print(f"总收益率: {ret_val:.4f}")
    print(f"最大回撤: {md_val:.4f}")
    print(f"交易次数: {n_trades}")

    if save_charts:
        os.makedirs(REPORT_DIR, exist_ok=True)
        safe_symbol = symbol.replace(".", "_")
        try:
            # 1) VectorBT 组合图：净值 / 回撤 / 交易等（默认子图）
            fig_pf = pf.plot(settings=dict(width=1200, height=700))
            path_pf = os.path.join(REPORT_DIR, f"dow_zigzag_obv_pf_{safe_symbol}.html")
            fig_pf.write_html(path_pf)
            print(f"组合图已保存: {path_pf}")
        except Exception as e:
            try:
                fig_pf = pf[symbol].plot(settings=dict(width=1200, height=700))
                path_pf = os.path.join(REPORT_DIR, f"dow_zigzag_obv_pf_{safe_symbol}.html")
                fig_pf.write_html(path_pf)
                print(f"组合图已保存: {path_pf}")
            except Exception as e2:
                print(f"组合图保存失败: {e2}")
        # 2) 价格 + ZigZag(MT4) + OBV + 入场/出场 标记
        pivots = zigzag_pivots_mt4(
            high, low, close,
            depth=ZZ_MT4_DEPTH,
            deviation=ZZ_MT4_DEVIATION,
            backstep=ZZ_MT4_BACKSTEP,
        )
        obv_series = calc_obv(pd.Series(close), pd.Series(volume))
        obv_arr = obv_series.values if hasattr(obv_series, "values") else np.asarray(obv_series)
        fig_price = _plot_price_zigzag_signals(
            index, close, pivots, entry_123, exit_123, symbol,
            title_suffix=f"| {obv_label} | {entry_label}",
            obv=obv_arr,
        )
        if fig_price is not None:
            path_price = os.path.join(REPORT_DIR, f"dow_zigzag_obv_price_{safe_symbol}.html")
            fig_price.write_html(path_price)
            print(f"价格+ZigZag+信号图已保存: {path_price}")

    return pf


if __name__ == "__main__":
    use_obv = USE_OBV_FILTER and "--no-obv" not in sys.argv
    save_charts = "--no-chart" not in sys.argv
    entry_mode = "123"
    for i, a in enumerate(sys.argv[1:], start=1):
        if a == "--entry" and i < len(sys.argv) - 1:
            entry_mode = sys.argv[i + 1].strip().lower()
            if entry_mode not in ("ma", "123"):
                entry_mode = "123"
            break
    symbol = "0700.HK"
    skip = False
    for a in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if a in ("--no-obv", "--entry"):
            if a == "--entry":
                skip = True
            continue
        if not a.startswith("--"):
            symbol = a
            break
    run_dow_zigzag_obv(symbol, use_obv_filter=use_obv, entry_mode=entry_mode, save_charts=save_charts)
