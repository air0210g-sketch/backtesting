import vectorbt as vbt
import pandas as pd
import numpy as np
import os
import itertools
import schedule
import time
from backtesting.notifier import TelegramNotifier
from backtesting.data_loader import load_data
import backtesting.indicators as inds


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

    def load_data(self, period_suffix="day"):
        """
        Load data from CSVs and align them.
        """
        print(f"Loading data from {self.data_dir}...")
        data_map = load_data(self.data_dir, period_suffix=period_suffix)
        if not data_map:
            print("No data loaded.")
            return False

        # Align Data (ffill)
        # Replace 0 with NaN first to avoid using 0 as valid price
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

        # Valid Mask (Close > 0)
        self.mask = (
            (self.close > 0) & (self.high > 0) & (self.low > 0) & self.close.notnull()
        )
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
        entries_shifted = entries.shift(1).fillna(False)
        exits_shifted = exits.shift(1).fillna(False)

        # Apply valid_open_mask if provided
        if "valid_open_mask" in kwargs:
            mask = kwargs.pop("valid_open_mask")
            # entries is (Time x Symbols), mask is (Time x Symbols)
            # direct & operation should work if columns align
            entries_shifted = entries_shifted & mask
            exits_shifted = exits_shifted & mask

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
                    k_v = f"{k[symbol].at[ts]:.2f}" if ts in k.index else "N/A"
                    d_v = f"{d[symbol].at[ts]:.2f}" if ts in d.index else "N/A"
                    j_v = f"{j[symbol].at[ts]:.2f}" if ts in j.index else "N/A"
                    wk_v = f"{wk[symbol].at[ts]:.2f}" if ts in wk.index else "N/A"
                    wd_v = f"{wd[symbol].at[ts]:.2f}" if ts in wd.index else "N/A"
                    wj_v = f"{wj[symbol].at[ts]:.2f}" if ts in wj.index else "N/A"

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
                trace.hovertemplate = "%{text}<br>价格: %{y:.2f}"
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
        **portfolio_kwargs,
    ):
        """
        Run Grid Search using vbt.IndicatorFactory (Vectorized).
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
        StrategyInd = vbt.IndicatorFactory(
            class_name="Strategy",
            short_name="st",
            input_names=["open_df", "high_df", "low_df", "close_df", "vol_df"],
            param_names=param_names,
            output_names=["entries", "exits"],
        ).from_apply_func(self.strategy_func, keep_pd=True)

        print(
            f"Running Signal Gen ({len(list(itertools.product(*sig_grid.values())))} combos) on FULL data..."
        )
        self.notify(
            f"🚀 开始优化，共有 {len(list(itertools.product(*sig_grid.values())))} 种参数组合。"
        )
        res = StrategyInd.run(
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            **sig_grid,
            param_product=True,
        )

        t_close = self.close.iloc[:split_idx]
        entries = res.entries.iloc[:split_idx]
        exits = res.exits.iloc[:split_idx]

        n_sig_combos = entries.shape[1] // t_close.shape[1]
        valid_mask_base = (t_close > 0) & t_close.notnull()
        valid_mask_sig = pd.DataFrame(
            np.tile(valid_mask_base.values, (1, n_sig_combos)),
            index=entries.index,
            columns=entries.columns,
        )

        entries = entries & valid_mask_sig
        exits = exits & valid_mask_sig

        # Combine checks for local updates of portfolio_kwargs if needed, or just merge
        # But here portfolio_kwargs is local dict, checking conflict with argument
        sim_kwargs = portfolio_kwargs.copy()
        portfolio_kwargs_local = {}
        all_param_names = param_names
        if len(exit_grid) > 0:
            exit_keys = list(exit_grid.keys())
            exit_values = list(exit_grid.values())
            exit_combos = list(itertools.product(*exit_values))
            n_exit_combos = len(exit_combos)

            print(f"Applying Exit Params ({n_exit_combos} combos)...")
            entries = pd.concat([entries] * n_exit_combos, axis=1)
            exits = pd.concat([exits] * n_exit_combos, axis=1)

            original_width = entries.shape[1] // n_exit_combos
            for i, k in enumerate(exit_keys):
                param_array = []
                for combo in exit_combos:
                    param_array.extend([combo[i]] * original_width)
                portfolio_kwargs_local[k] = np.array(param_array)

            original_cols = res.entries.columns
            new_col_tuples = []
            for combo in exit_combos:
                for col_tuple in original_cols:
                    new_col_tuples.append(col_tuple[:-1] + combo + col_tuple[-1:])

            entries.columns = pd.MultiIndex.from_tuples(
                new_col_tuples,
                names=res.entries.columns.names[:-1]
                + exit_keys
                + res.entries.columns.names[-1:],
            )
            exits.columns = entries.columns
            all_param_names = param_names + exit_keys

        t_open = self.open.iloc[:split_idx]
        entries_shifted = entries.shift(1).fillna(False)
        exits_shifted = exits.shift(1).fillna(False)

        # Safety: Ensure execution price (open) is valid
        valid_open = (t_open > 0) & t_open.notnull()
        # Broadcast valid_open to match entries_shifted shape if needed,
        # but vbt inputs are aligned.
        # entries_shifted has MultiIndex columns (params), valid_open has Index (symbols).
        # We need to align carefully.
        # Actually t_open is DataFrame (Time x Symbols).
        # entries_shifted is DataFrame (Time x (Symbols*Params)).
        # Simple boolean indexing might not work directly due to column mismatch.

        # Better approach: let vbt handle init_cash, but here we must fix the signal.
        # Ensure that for each column in entries_shifted, the corresponding symbol's price is valid.

        # We can reconstruct the symbol mapping from columns, or just trust vbt
        # BUT vbt crashes.

        # Quick fix: Vectorized check using underlying values if shapes match in time.
        # But columns don't match.

        # However, vbt signals logic usually aligns.
        # Just filling NaNs in t_open with 0 and check > 0.

        # Actually, let's just use vbt's built-in alignment if possible,
        # or iterate? No, iteration is slow.

        # The issue is probably leading NaNs.
        # t_open might have NaNs at start.
        # Shifted signal might land on NaN.

        # Let's use 0.0 for NaNs in price passed to vbt? No, price must be > 0.

        # We need to zero out signals where price is invalid.
        # valid_open is (Time x Symbols).
        # entries_shifted columns are MultiIndex, where one level is symbol?
        # Let's check column structure.
        # run_optimization builds columns: param_names.
        # Wait, entries from StrategyInd.run has columns corresponding to params?
        # StrategyInd.run(..., param_product=True) returns entries with columns = params combination?
        # AND it is broadcast over symbols?
        # If input was DataFrame (Time x Symbols), output entries is (Time x (Symbols * Params)).

        # Correct. The columns are MultiIndex: (Param1, Param2, ..., Symbol).
        # Or (Symbol, Param1, ...)?
        # vbt typically puts Symbol as last level or first?
        # Usually StrategyInd.run with DataFrame inputs returns specific shape.

        # Let's look at `valid_mask_sig` creation:
        # valid_mask_base = (t_close > 0) & t_close.notnull()
        # valid_mask_sig = pd.DataFrame(np.tile(valid_mask_base.values, (1, n_sig_combos)), ...)

        # I can do the same for t_open.
        valid_open_base = (t_open > 0) & t_open.notnull()
        # Reuse n_sig_combos calculation logic or just re-calculate
        # n_sig_combos was entries.shape[1] // t_close.shape[1]
        # But wait, we added exit combos, modifying entries.shape[1].

        # Current entries has shape (Time x (SignCombos * ExitCombos * Symbols)).
        # We need to tile valid_open_base to match.

        # Recalculate tiling factor
        n_total_cols = entries_shifted.shape[1]
        n_symbols = t_open.shape[1]
        n_reps = n_total_cols // n_symbols

        if n_reps > 0:
            valid_open_sig = np.tile(valid_open_base.values, (1, n_reps))
            entries_shifted = entries_shifted & valid_open_sig
            exits_shifted = exits_shifted & valid_open_sig

        pf = vbt.Portfolio.from_signals(
            close=t_close,
            entries=entries_shifted,
            exits=exits_shifted,
            price=t_open,
            init_cash=1000000,
            fees=0.0003,
            slippage=0.0005,
            freq="1D",
            **portfolio_kwargs_local,
            **sim_kwargs,
        )

        level_indices = list(range(len(all_param_names)))
        total_ret = pf.total_return().groupby(level=level_indices).mean()
        max_dd = pf.max_drawdown().groupby(level=level_indices).mean()
        win_rate = pf.trades.win_rate().groupby(level=level_indices).mean()
        sharpe = pf.sharpe_ratio().groupby(level=level_indices).mean()

        results_df = pd.DataFrame(
            {
                "return": total_ret,
                "max_dd": max_dd,
                "win_rate": win_rate,
                "sharpe": sharpe,
            }
        )
        results_df["calmar"] = results_df.apply(
            lambda row: row["return"] / abs(row["max_dd"]) if row["max_dd"] != 0 else 0,
            axis=1,
        )

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
        print(f"Metrics:\n{best_metrics[['return', 'max_dd', 'win_rate', 'calmar']]}")

        self.notify(
            f"✅ 优化完成。\n"
            f"最佳参数: {best_params}\n"
            f"回报率: {best_metrics['return']:.2%}\n"
            f"最大回撤: {best_metrics['max_dd']:.2%}\n"
            f"夏普比率: {best_metrics['sharpe']:.2f}\n"
            f"卡玛比率: {best_metrics['calmar']:.2f}"
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
        v_close = self.close.iloc[split_idx:]
        entries = all_entries.iloc[split_idx:]
        exits = all_exits.iloc[split_idx:]
        v_mask = self.mask.iloc[split_idx:]
        entries &= v_mask
        exits &= v_mask

        EXIT_KEYS = ["sl_stop", "tp_stop", "sl_trail"]
        exit_params = {k: v for k, v in best_params.items() if k in EXIT_KEYS}
        v_open = self.open.iloc[split_idx:]

        # Fix: Filter out signals where execution price (Open) is 0 or NaN
        valid_open = (v_open > 0) & v_open.notnull()
        pf, exits_shifted = self._run_single_backtest(
            v_close,
            v_open,
            entries,
            exits,
            valid_open_mask=valid_open,
            **exit_params,
            **portfolio_kwargs,
        )
        val_metrics = self._print_stats(pf, "Validation_Set")
        pf.stats().to_csv(os.path.join(self.report_dir, "validation_stats.csv"))
        return pf, val_metrics, exits_shifted

    def _print_stats(self, portfolio, name):
        tr = portfolio.total_return()
        mean_ret = tr.mean()
        win_rate = portfolio.trades.win_rate().mean()
        max_dd = portfolio.max_drawdown().mean()
        sharpe = portfolio.sharpe_ratio().mean()
        calmar = mean_ret / abs(max_dd) if max_dd != 0 else 0
        print(
            f"[{name}]\nReturn: {mean_ret*100:.2f}%, WinRate: {win_rate*100:.2f}%, MaxDD: {max_dd*100:.2f}%, Sharpe: {sharpe:.2f}, Calmar: {calmar:.2f}"
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
        print(comparison)

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
            f"验证集回报: {val_metrics['return']:.2%}\n"
            f"验证集最大回撤: {val_metrics['max_dd']:.2%}\n"
            f"验证集夏普: {val_metrics['sharpe']:.2f}"
        )

        keys = list(best_params.keys())
        if len(keys) >= 2:
            self.plot_param_heatmap(keys[0], keys[1], metric=target_metric)
        return best_params, val_metrics
