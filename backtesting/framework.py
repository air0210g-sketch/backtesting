import warnings

# vectorbt 多列聚合时的 mean 会触发 pandas UserWarning，此处统一忽略
warnings.filterwarnings("ignore", message="Object has multiple columns.*Aggregating using.*mean")

import vectorbt as vbt
import pandas as pd
import numpy as np
import os
import itertools
import schedule
import time
import json
from typing import Any, Callable, Dict, Optional

from backtesting.notifier import TelegramNotifier
from backtesting.data_loader import load_data
import backtesting.indicators as inds


# 预计算指标默认实现：策略中可注入 k_df/d_df/j_df、atr_df 等，只算一次
def _precompute_kdj(o, h, l, c, v):
    """日线 KDJ (N=9, M=3) -> k_df, d_df, j_df"""
    k, d, j = inds.calc_kdj(c, h, l, N=9, M=3)
    return {"k_df": k, "d_df": d, "j_df": j}


def _precompute_weekly_kdj(o, h, l, c, v):
    """周线 KDJ 对齐日线 (N=9, M=3) -> k_df, d_df, j_df"""
    wk, wd, wj = inds.calc_weekly_kdj(o, h, l, c, N=9, M=3)
    return {"k_df": wk, "d_df": wd, "j_df": wj}


def _precompute_atr(o, h, l, c, v, period=14):
    """ATR (默认 14) -> atr_df"""
    return {"atr_df": inds.calc_atr(h, l, c, period=period)}


# 注册表：name -> (open, high, low, close, volume) -> dict of DataFrame
# 新增指标可在此扩展，策略中对应增加 xxx_df 参数即可
PRECOMPUTE_REGISTRY = {
    "kdj": _precompute_kdj,
    "weekly_kdj": _precompute_weekly_kdj,
    "atr": lambda o, h, l, c, v: _precompute_atr(o, h, l, c, v, period=14),
}
# 默认启用项（仅周 KDJ，与常用策略一致；可按需改为 ["kdj"] 或 ["kdj", "atr"]）
DEFAULT_PRECOMPUTE_NAMES = ["weekly_kdj"]


def _safe_prices(df: pd.DataFrame) -> pd.DataFrame:
    """确保价格有限且 >0，满足 vectorbt 下单要求。"""
    out = df.astype(float)
    out = out.where(np.isfinite(out) & (out > 0), np.nan)
    out = out.ffill().bfill()
    out = out.where(out.notna(), 1.0)
    return out.astype(float)


# 止损/止盈键名，用于从列 MultiIndex 提取 per-column 数组（VectorBT 广播）
_EXIT_PARAM_KEYS = ("sl_stop", "tp_stop", "sl_trail")


def _run_optimization_chunk(args):
    (
        entries_chunk,
        exits_chunk,
        t_close,
        t_open,
        exit_grid,
        param_names,
        from_signals_kwargs,
    ) = args
    import itertools as _it

    if len(exit_grid) > 0:
        exit_keys = list(exit_grid.keys())
        exit_values = list(exit_grid.values())
        exit_combos = list(_it.product(*exit_values))
        n_exit_combos = len(exit_combos)
        entries_expanded = pd.concat([entries_chunk] * n_exit_combos, axis=1)
        exits_expanded = pd.concat([exits_chunk] * n_exit_combos, axis=1)
        orig_cols = entries_chunk.columns
        new_col_tuples = []
        for combo in exit_combos:
            for col_tuple in orig_cols:
                new_col_tuples.append(col_tuple[:-1] + combo + col_tuple[-1:])
        new_idx = pd.MultiIndex.from_tuples(
            new_col_tuples,
            names=entries_chunk.columns.names[:-1]
            + exit_keys
            + entries_chunk.columns.names[-1:],
        )
        entries_expanded.columns = new_idx
        exits_expanded.columns = new_idx
    else:
        entries_expanded = entries_chunk
        exits_expanded = exits_chunk

    shifted_e = entries_expanded.shift(1)
    shifted_x = exits_expanded.shift(1)
    entries_shifted = shifted_e.where(shifted_e.notna(), False).astype(bool)
    exits_shifted = shifted_x.where(shifted_x.notna(), False).astype(bool)

    # 按列广播止损/止盈参数（VectorBT 支持 per-column 数组，避免标量重复）
    kwargs = dict(from_signals_kwargs)
    for key in _EXIT_PARAM_KEYS:
        if key in entries_expanded.columns.names:
            level_ix = entries_expanded.columns.names.index(key)
            kwargs[key] = entries_expanded.columns.get_level_values(level_ix).astype(
                float
            ).values

    pf = vbt.Portfolio.from_signals(
        close=t_close,
        entries=entries_shifted,
        exits=exits_shifted,
        price=t_open,
        **kwargs,
    )

    lvl_indices = list(range(len(entries_expanded.columns.names) - 1))
    total_ret = pf.total_return().groupby(level=lvl_indices).mean()
    max_dd = pf.max_drawdown().groupby(level=lvl_indices).mean()
    win_rate = pf.trades.win_rate().groupby(level=lvl_indices).mean()
    sharpe = pf.sharpe_ratio().groupby(level=lvl_indices).mean()
    return pd.DataFrame(
        {
            "return": total_ret,
            "max_dd": max_dd,
            "win_rate": win_rate,
            "sharpe": sharpe,
        }
    )


class BacktestRunner:
    def __init__(
        self, data_dir, report_dir="reports", telegram_token=None, telegram_chat_id=None
    ):
        self.data_dir = data_dir
        self.report_dir = report_dir
        self.notifier = TelegramNotifier(token=telegram_token, chat_id=telegram_chat_id)

        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)

        self.close = None
        self.high = None
        self.low = None
        self.open = None
        self.volume = None

        self.strategy_func = None
        # 预计算指标：name -> (open, high, low, close, vol) -> DataFrame 或 dict of DataFrame
        self.precompute_indicators = None

    def set_precompute_indicators(
        self,
        indicators: Optional[Any] = None,
    ):
        """
        设置预计算指标，固定周期指标只算一次，避免在网格内重复计算。

        支持三种用法：
        - 字符串 "default"：使用内置默认（当前为 weekly_kdj）。
        - 名字列表：如 ["kdj", "weekly_kdj", "atr"]，从 PRECOMPUTE_REGISTRY 取实现。
        - 自定义 dict：{ "名字": fn }，fn(open, high, low, close, volume) -> DataFrame 或 dict of DataFrame。

        策略函数需声明对应参数（如 k_df, d_df, j_df 或 atr_df），有注入时直接使用，否则在策略内计算。
        """
        if indicators is None:
            self.precompute_indicators = None
            return
        if indicators == "default":
            names = DEFAULT_PRECOMPUTE_NAMES
            self.precompute_indicators = {n: PRECOMPUTE_REGISTRY[n] for n in names if n in PRECOMPUTE_REGISTRY}
            return
        if isinstance(indicators, (list, tuple)):
            self.precompute_indicators = {
                n: PRECOMPUTE_REGISTRY[n]
                for n in indicators
                if n in PRECOMPUTE_REGISTRY
            }
            if len(self.precompute_indicators) < len(indicators):
                unknown = set(indicators) - set(PRECOMPUTE_REGISTRY)
                if unknown:
                    print(f"预计算未知项（已忽略）: {unknown}，可选: {list(PRECOMPUTE_REGISTRY.keys())}")
            return
        self.precompute_indicators = indicators

    def notify(self, message):
        """Send a notification via Telegram."""
        if self.notifier:
            self.notifier.send_message(message)

    def run_scheduled_job(
        self, job_func, interval_unit="minutes", interval_value=1, **kwargs
    ):
        """
        Run a job periodically.
        interval_unit: 'seconds', 'minutes', 'hours', 'days'
        interval_value: integer
        """
        print(f"Scheduling job every {interval_value} {interval_unit}...")

        if interval_unit == "seconds":
            schedule.every(interval_value).seconds.do(job_func, **kwargs)
        elif interval_unit == "minutes":
            schedule.every(interval_value).minutes.do(job_func, **kwargs)
        elif interval_unit == "hours":
            schedule.every(interval_value).hours.do(job_func, **kwargs)
        elif interval_unit == "days":
            schedule.every(interval_value).days.do(job_func, **kwargs)
        else:
            raise ValueError(f"Unsupported interval unit: {interval_unit}")

        while True:
            schedule.run_pending()
            time.sleep(1)

    def load_data(self, period_suffix="day", use_cache=True):
        """
        从 CSV 加载并对齐数据。use_cache=True 时将对齐结果缓存为 .parquet，
        下次若数据目录下 CSV 未变则直接读缓存，避免重复扫描与对齐。
        """
        cache_dir = os.path.join(self.data_dir, ".cache")
        manifest_path = os.path.join(cache_dir, "_manifest.json")

        def _csv_list_with_mtime():
            out = []
            for f in os.listdir(self.data_dir):
                if f.endswith(".csv") and (not period_suffix or f.endswith(f"_{period_suffix}.csv")):
                    path = os.path.join(self.data_dir, f)
                    if os.path.isfile(path):
                        out.append((f, os.path.getmtime(path)))
            return sorted(out)

        if use_cache and os.path.isdir(cache_dir) and os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                current = _csv_list_with_mtime()
                saved = [tuple(x) for x in manifest.get("files", [])]
                if manifest.get("period_suffix") == period_suffix and saved == current:
                    print(f"Loading aligned data from cache ({cache_dir})...")
                    try:
                        self.open = pd.read_parquet(os.path.join(cache_dir, "open.parquet"))
                        self.high = pd.read_parquet(os.path.join(cache_dir, "high.parquet"))
                        self.low = pd.read_parquet(os.path.join(cache_dir, "low.parquet"))
                        self.close = pd.read_parquet(os.path.join(cache_dir, "close.parquet"))
                        self.volume = pd.read_parquet(os.path.join(cache_dir, "volume.parquet"))
                    except Exception:
                        with open(os.path.join(cache_dir, "ohlcv.pkl"), "rb") as f:
                            import pickle
                            data = pickle.load(f)
                        self.open, self.high, self.low, self.close, self.volume = (
                            data["open"], data["high"], data["low"], data["close"], data["volume"]
                        )
                    self.mask = (
                        (self.close > 0)
                        & (self.high > 0)
                        & (self.low > 0)
                        & self.close.notnull()
                    )
                    print(f"Cached load: {len(self.close.columns)} symbols.")
                    return True
            except Exception as e:
                print(f"Cache read failed ({e}), falling back to CSV.")

        print(f"Loading data from {self.data_dir}...")
        data_map = load_data(self.data_dir, period_suffix=period_suffix)
        if not data_map:
            print("No data loaded.")
            return False

        print(f"Aligning {len(data_map)} symbols...")
        self.close = (
            pd.DataFrame({sym: df["close"] for sym, df in data_map.items()})
            .replace(0, np.nan)
            .ffill()
        )
        self.open = (
            pd.DataFrame({sym: df["open"] for sym, df in data_map.items()})
            .replace(0, np.nan)
            .ffill()
        )
        self.high = (
            pd.DataFrame({sym: df["high"] for sym, df in data_map.items()})
            .replace(0, np.nan)
            .ffill()
        )
        self.low = (
            pd.DataFrame({sym: df["low"] for sym, df in data_map.items()})
            .replace(0, np.nan)
            .ffill()
        )
        self.volume = (
            pd.DataFrame({sym: df["volume"] for sym, df in data_map.items()})
            .replace(0, np.nan)
            .ffill()
        )
        self.mask = (
            (self.close > 0) & (self.high > 0) & (self.low > 0) & self.close.notnull()
        )

        if use_cache:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                try:
                    self.open.to_parquet(os.path.join(cache_dir, "open.parquet"))
                    self.high.to_parquet(os.path.join(cache_dir, "high.parquet"))
                    self.low.to_parquet(os.path.join(cache_dir, "low.parquet"))
                    self.close.to_parquet(os.path.join(cache_dir, "close.parquet"))
                    self.volume.to_parquet(os.path.join(cache_dir, "volume.parquet"))
                except Exception:
                    import pickle
                    with open(os.path.join(cache_dir, "ohlcv.pkl"), "wb") as f:
                        pickle.dump({
                            "open": self.open, "high": self.high, "low": self.low,
                            "close": self.close, "volume": self.volume,
                        }, f)
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"period_suffix": period_suffix, "files": _csv_list_with_mtime()},
                        f,
                        indent=0,
                    )
                print(f"Cache written to {cache_dir}.")
            except Exception as e:
                print(f"Cache write failed: {e}")

        return True

    def set_strategy(self, func):
        """
        Set the strategy logic function.
        Function signature: func(open, high, low, close, volume, **kwargs) -> (entries, exits)
        entries, exits should be boolean DataFrames/Series.
        """
        self.strategy_func = func

    def _run_single_backtest(self, close_data, open_data, entries, exits, **kwargs):
        """
        Run a single vectorbt portfolio simulation with T+1 Open execution.
        Manually shifts signals for compatibility.
        """
        # Manual Shift: T-day signal -> T+1 day execution
        se, sx = entries.shift(1), exits.shift(1)
        entries_shifted = se.where(se.notna(), False).astype(bool)
        exits_shifted = sx.where(sx.notna(), False).astype(bool)

        # Apply valid_open_mask if provided
        if "valid_open_mask" in kwargs:
            # Just pop it to avoid passing to vbt, logic handling removed for perf
            # (Reliance on load_data cleaning)
            kwargs.pop("valid_open_mask")

        pf = vbt.Portfolio.from_signals(
            close=close_data,
            entries=entries_shifted,
            exits=exits_shifted,
            price=open_data,  # Pass data directly
            init_cash=1000000,
            fees=0.0003,
            slippage=0.0005,
            freq="1D",
            **kwargs,
        )
        return pf, exits_shifted

    def analyze_trades(
        self, pf, exits_mask=None, sl_stop=None, tp_stop=None, sl_trail=None
    ):
        """
        Enrich trade records with Exit Reason (Signal, Stop Loss, Trailing Stop, etc.).
        Cross-references exits_mask if provided.
        """
        trades = pf.trades.records_readable

        # Add Reason Column
        trades["Exit Reason"] = "Signal/Force"  # Default

        for i, row in trades.iterrows():
            ts = row["Exit Timestamp"]
            col = row["Column"]
            ret = row["Return"]

            # 1. Check if it was a signal exit
            is_signal = False
            if exits_mask is not None:
                try:
                    if ts in exits_mask.index:
                        val = exits_mask.at[ts, col]
                        if isinstance(val, (pd.Series, np.ndarray)):
                            is_signal = any(val)
                        else:
                            is_signal = bool(val)
                except Exception:
                    is_signal = False

            # 2. If not a signal, it's a stop
            if not is_signal:
                if sl_stop and ret <= -sl_stop * 0.95:
                    trades.at[i, "Exit Reason"] = "Stop Loss"
                elif tp_stop and ret >= tp_stop * 0.95:
                    trades.at[i, "Exit Reason"] = "Take Profit"
                elif sl_trail:
                    # Trailing Stop is the likely culprit if not hard SL or Signal
                    trades.at[i, "Exit Reason"] = "Trailing Stop"
                else:
                    trades.at[i, "Exit Reason"] = "Stop/Force"

        print("\n--- Trade Attribution Analysis ---")
        print(trades["Exit Reason"].value_counts())
        return trades

    def plot_trades(self, pf, symbol=None, trade_logs=None):
        """
        Plot trades for a specific symbol using vectorbt's interactive Plotly.
        Overlay 'Exit Reason', KDJ, Weekly KDJ, Volume and Chinese Date.
        """
        if symbol is None:
            symbol = pf.wrapper.columns[0]  # Pick first one

        print(f"\nPlotting trades for {symbol}...")

        # 1. Calculate Indicators for this specific symbol for plotting
        n, m = 9, 3
        k, d, j = inds.calc_kdj(
            self.close[[symbol]], self.high[[symbol]], self.low[[symbol]], n, m
        )
        wk, wd, wj = inds.calc_weekly_kdj(
            self.open[[symbol]],
            self.high[[symbol]],
            self.low[[symbol]],
            self.close[[symbol]],
            n,
            m,
        )
        vol = self.volume[symbol]

        # 2. Setup Base Plot
        pf_single = pf[symbol]
        fig = pf_single.plot(settings=dict(width=1200, height=600))

        # 3. Enhance Hover Info
        exit_map = {}
        if trade_logs is not None:
            subset = trade_logs[trade_logs["Column"] == symbol].copy()
            if not subset.empty:
                subset["Exit Timestamp"] = pd.to_datetime(subset["Exit Timestamp"])
                exit_map = subset.set_index("Exit Timestamp")["Exit Reason"].to_dict()

        for trace in fig.data:
            if trace.name in ["Entries", "Exits", "Buy", "Sell", "Exit"]:
                new_text = []
                for t in trace.x:
                    ts = pd.Timestamp(t)
                    v_val = f"{vol.at[ts]:,.0f}" if ts in vol.index else "N/A"
                    date_zh = ts.strftime("%Y年%m月%d日")
                    k_v = f"{k[symbol].at[ts]:.3f}" if ts in k.index else "N/A"
                    d_v = f"{d[symbol].at[ts]:.3f}" if ts in d.index else "N/A"
                    j_v = f"{j[symbol].at[ts]:.3f}" if ts in j.index else "N/A"
                    wk_v = f"{wk[symbol].at[ts]:.3f}" if ts in wk.index else "N/A"
                    wd_v = f"{wd[symbol].at[ts]:.3f}" if ts in wd.index else "N/A"
                    wj_v = f"{wj[symbol].at[ts]:.3f}" if ts in wj.index else "N/A"

                    info = (
                        f"<b>{date_zh}</b><br>"
                        f"成交量: {v_val}<br>"
                        f"KDJ: {k_v} / {d_v} / {j_v}<br>"
                        f"周KDJ: {wk_v} / {wd_v} / {wj_v}"
                    )

                    if "Exit" in trace.name or "Sell" in trace.name:
                        reason = "Signal/Force"
                        if ts in exit_map:
                            reason = exit_map[ts]
                        info += f"<br>离场原因: {reason}"

                    new_text.append(info)

                trace.text = tuple(new_text)
                trace.hovertemplate = "%{text}<br>价格: %{y:.3f}"
                print(f"Enhanced {trace.name} trace with detailed info.")

        plot_symbol_str = str(symbol) if isinstance(symbol, str) else str(symbol[-1])
        plot_path = os.path.join(self.report_dir, f"trade_plot_{plot_symbol_str}.html")
        fig.write_html(plot_path)
        print(f"Saved interactive plot to {plot_path}")
        return plot_path

    def run_optimization(
        self,
        param_grid,
        split_ratio=0.7,
        target_metric="calmar",
        chunk_size_combos=50,
        n_jobs=1,
        sample_max=None,
        sample_seed=42,
        **portfolio_kwargs,
    ):
        """
        Run Grid Search using vbt.IndicatorFactory (Vectorized)。

        可选优化：chunk_size_combos（每批组合数）、n_jobs（并行批次数）、sample_max（随机抽样组合数）。
        """
        if self.strategy_func is None:
            raise ValueError("Strategy function not set. Use set_strategy().")

        EXIT_KEYS = ["sl_stop", "tp_stop", "sl_trail"]
        sig_grid = {k: v for k, v in param_grid.items() if k not in EXIT_KEYS}
        exit_grid = {k: v for k, v in param_grid.items() if k in EXIT_KEYS}

        split_idx = int(len(self.close) * split_ratio)
        print(
            f"Splitting Data: Train ({split_idx}) | Validation ({len(self.close)-split_idx})"
        )

        param_names = list(sig_grid.keys())
        base_inputs = ["open_df", "high_df", "low_df", "close_df", "vol_df"]
        precomputed = {}
        if self.precompute_indicators:
            print("Precomputing fixed indicators (once)...")
            for name, fn in self.precompute_indicators.items():
                out = fn(
                    self.open,
                    self.high,
                    self.low,
                    self.close,
                    self.volume,
                )
                if isinstance(out, dict):
                    precomputed.update(out)
                else:
                    precomputed[name] = out
            print(f"  Precomputed: {list(precomputed.keys())}")

        input_names = base_inputs + list(precomputed.keys())
        StrategyInd = vbt.IndicatorFactory(
            class_name="Strategy",
            short_name="st",
            input_names=input_names,
            param_names=param_names,
            output_names=["entries", "exits"],
        ).from_apply_func(self.strategy_func, keep_pd=True)

        n_sig_combos = len(list(itertools.product(*sig_grid.values())))
        print(f"Running Signal Gen ({n_sig_combos} combos) on FULL data...")
        self.notify(f"🚀 开始优化，共有 {n_sig_combos} 种参数组合。")
        res = StrategyInd.run(
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            **precomputed,
            **sig_grid,
            param_product=True,
        )

        t_close = self.close.iloc[:split_idx]
        entries = res.entries.iloc[:split_idx]
        exits = res.exits.iloc[:split_idx]

        # Ensure price/close are finite and > 0 (vectorbt requirement for order execution)
        t_close = _safe_prices(t_close)
        t_open = self.open.iloc[:split_idx]
        t_open = _safe_prices(t_open)

        sim_kwargs = portfolio_kwargs.copy()
        portfolio_kwargs_local = {}
        all_param_names = param_names

        n_symbols = len(t_close.columns)
        total_cols = entries.shape[1]
        n_combos = total_cols // n_symbols

        # 可选：仅对信号参数组合做随机抽样
        if sample_max is not None and n_combos > sample_max:
            rng = np.random.default_rng(sample_seed)
            selected = rng.choice(n_combos, size=sample_max, replace=False)
            selected.sort()
            col_ix = np.concatenate(
                [np.arange(c * n_symbols, (c + 1) * n_symbols) for c in selected]
            )
            entries = entries.iloc[:, col_ix]
            exits = exits.iloc[:, col_ix]
            total_cols = entries.shape[1]
            print(f"Sampled {sample_max} signal combos (from {n_combos}), columns={total_cols}")

        cols_per_chunk = chunk_size_combos * n_symbols
        from_signals_kwargs = {
            "init_cash": 1000000,
            "fees": 0.0003,
            "slippage": 0.0005,
            "freq": "1D",
            **portfolio_kwargs_local,
            **sim_kwargs,
        }

        import gc

        chunk_arg_list = []
        for i in range(0, total_cols, cols_per_chunk):
            end_i = min(i + cols_per_chunk, total_cols)
            chunk_arg_list.append(
                (
                    entries.iloc[:, i:end_i],
                    exits.iloc[:, i:end_i],
                    t_close,
                    t_open,
                    exit_grid,
                    param_names,
                    from_signals_kwargs,
                )
            )

        if n_jobs is not None and n_jobs != 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            n_workers = n_jobs if n_jobs > 0 else max(1, (os.cpu_count() or 1))
            chunk_results = []
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futs = {
                    executor.submit(_run_optimization_chunk, a): idx
                    for idx, a in enumerate(chunk_arg_list)
                }
                for fut in as_completed(futs):
                    idx = futs[fut]
                    try:
                        chunk_results.append(fut.result())
                        print(f"Batch {idx + 1}/{len(chunk_arg_list)} done.")
                    except Exception as e:
                        print(f"Batch {idx + 1} error: {e}")
        else:
            chunk_results = []
            for idx, args in enumerate(chunk_arg_list):
                print(f"Processing batch {idx + 1}/{len(chunk_arg_list)}...")
                try:
                    chunk_results.append(_run_optimization_chunk(args))
                except Exception as e:
                    print(f"Metrics calc error in batch: {e}")
                gc.collect()

        if not chunk_results:
            print("No results generated.")
            return {}, pd.DataFrame()

        results_df = pd.concat(chunk_results)
        results_df["calmar"] = results_df.apply(
            lambda row: row["return"] / abs(row["max_dd"]) if row["max_dd"] != 0 else 0,
            axis=1,
        )

        # Update all_param_names for final usage
        if len(exit_grid) > 0:
            all_param_names = param_names + list(exit_grid.keys())
        else:
            all_param_names = param_names

        best_idx = results_df[target_metric].idxmax()
        best_params = (
            {all_param_names[0]: best_idx}
            if len(all_param_names) == 1
            else dict(zip(all_param_names, best_idx))
        )

        # Cast types for best_params
        for k, v_list in param_grid.items():
            if v_list and isinstance(v_list[0], int):
                try:
                    best_params[k] = int(best_params[k])
                except Exception:
                    pass

        best_metrics = results_df.loc[best_idx]
        print(
            f"\nOptimization Completed. Best Params ({target_metric}):\n{best_params}"
        )
        m = best_metrics
        print(
            f"Metrics:\n"
            f"  return:   {m['return']:.3f}\n"
            f"  max_dd:   {m['max_dd']:.3f}\n"
            f"  win_rate: {m['win_rate']:.3f}\n"
            f"  calmar:   {m['calmar']:.3f}"
        )

        self.notify(
            f"✅ 优化完成。\n"
            f"最佳参数: {best_params}\n"
            f"回报率: {best_metrics['return']:.3%}\n"
            f"最大回撤: {best_metrics['max_dd']:.3%}\n"
            f"夏普比率: {best_metrics['sharpe']:.3f}\n"
            f"卡玛比率: {best_metrics['calmar']:.3f}"
        )

        results_df.index.names = all_param_names
        self.results_df = results_df
        results_df.to_csv(os.path.join(self.report_dir, "optimization_results.csv"))
        return best_params, best_metrics

    def plot_param_heatmap(self, x_param, y_param, metric="calmar"):
        if not hasattr(self, "results_df") or self.results_df is None:
            return
        try:
            df = self.results_df.reset_index()
            pivot_data = df.groupby([x_param, y_param])[metric].mean().reset_index()
            pivot_table = pivot_data.pivot(
                index=y_param, columns=x_param, values=metric
            )
            import plotly.graph_objects as go

            fig = go.Figure(
                data=go.Heatmap(
                    z=pivot_table.values,
                    x=pivot_table.columns,
                    y=pivot_table.index,
                    colorscale="Viridis",
                    colorbar=dict(title=metric),
                )
            )
            fig.update_layout(
                title=f"Plateau: {x_param} vs {y_param} ({metric})",
                xaxis_title=x_param,
                yaxis_title=y_param,
                width=800,
                height=600,
            )
            path = os.path.join(
                self.report_dir, f"heatmap_{x_param}_{y_param}_{metric}.html"
            )
            fig.write_html(path)
            print(f"Saved Heatmap to {path}")
            return path
        except Exception as e:
            print(f"Heatmap Failed: {e}")

    def run_validation(self, best_params, split_ratio=0.7, **portfolio_kwargs):
        split_idx = int(len(self.close) * split_ratio)
        print("\nRunning Validation (Out-of-Sample) with Full-Data Signals...")
        all_entries, all_exits = self.strategy_func(
            self.open, self.high, self.low, self.close, self.volume, **best_params
        )
        v_close = _safe_prices(self.close.iloc[split_idx:])
        v_open = _safe_prices(self.open.iloc[split_idx:])
        entries = all_entries.iloc[split_idx:]
        exits = all_exits.iloc[split_idx:]
        v_mask = self.mask.iloc[split_idx:]
        entries &= v_mask
        exits &= v_mask

        EXIT_KEYS = ["sl_stop", "tp_stop", "sl_trail"]
        exit_params = {k: v for k, v in best_params.items() if k in EXIT_KEYS}

        pf, exits_shifted = self._run_single_backtest(
            v_close,
            v_open,
            entries,
            exits,
            **exit_params,
            **portfolio_kwargs,
        )
        val_metrics = self._print_stats(pf, "Validation_Set")
        pf.stats().to_csv(os.path.join(self.report_dir, "validation_stats.csv"))
        return pf, val_metrics, exits_shifted

    def _print_stats(self, portfolio, name):
        tr = np.asarray(portfolio.total_return()).ravel()
        mean_ret = float(tr.mean())
        win_rate = float(np.asarray(portfolio.trades.win_rate()).ravel().mean())
        max_dd = float(np.asarray(portfolio.max_drawdown()).ravel().mean())
        sharpe = float(np.asarray(portfolio.sharpe_ratio()).ravel().mean())
        calmar = mean_ret / abs(max_dd) if max_dd != 0 else 0
        print(
            f"[{name}]\nReturn: {mean_ret*100:.3f}%, WinRate: {win_rate*100:.3f}%, MaxDD: {max_dd*100:.3f}%, Sharpe: {sharpe:.3f}, Calmar: {calmar:.3f}"
        )
        return pd.Series(
            {
                "return": mean_ret,
                "max_dd": max_dd,
                "win_rate": win_rate,
                "sharpe": sharpe,
                "calmar": calmar,
            }
        )

    def run_full_cycle(
        self,
        strategy_func,
        param_grid,
        split_ratio=0.7,
        target_metric="calmar",
        **portfolio_kwargs,
    ):
        self.set_strategy(strategy_func)
        best_params, train_metrics = self.run_optimization(
            param_grid, split_ratio, target_metric, **portfolio_kwargs
        )
        val_pf, val_metrics, val_exits_shifted = self.run_validation(
            best_params, split_ratio, **portfolio_kwargs
        )

        print("\n=== TRAIN vs TEST COMPARISON ===")
        comparison = pd.DataFrame({"Train": train_metrics, "Test": val_metrics})
        comparison["Diff%"] = (
            (comparison["Test"] - comparison["Train"]) / comparison["Train"].abs() * 100
        )
        print(comparison.round(3))

        exit_params = {
            k: best_params[k]
            for k in ["sl_stop", "tp_stop", "sl_trail"]
            if k in best_params
        }
        trade_logs = self.analyze_trades(
            val_pf, exits_mask=val_exits_shifted, **exit_params
        )
        trade_logs.to_csv(os.path.join(self.report_dir, "trade_attribution.csv"))

        if len(trade_logs) > 0:
            top_symbol = trade_logs["Column"].value_counts().index[0]
            plot_path = self.plot_trades(
                val_pf, symbol=top_symbol, trade_logs=trade_logs
            )
            if self.notifier:
                self.notifier.send_message(f"📊 正在发送 {top_symbol} 的交易图表...")
                self.notifier.send_image(plot_path, caption=f"交易分析: {top_symbol}")

        self.notify(
            f"🚀 全流程完成。\n"
            f"验证集回报: {val_metrics['return']:.3%}\n"
            f"验证集最大回撤: {val_metrics['max_dd']:.3%}\n"
            f"验证集夏普: {val_metrics['sharpe']:.3f}"
        )

        keys = list(best_params.keys())
        if len(keys) >= 2:
            self.plot_param_heatmap(keys[0], keys[1], metric=target_metric)
        return best_params, val_metrics
