import vectorbt as vbt
import pandas as pd
import numpy as np
import os
import sys

# Add current path to import data_loader
# Add parent path to import data_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtesting.data_loader import load_data, align_and_clean_data

# Report Directory
REPORT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"
)
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock_data"
)
if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)


def apply_signal_logic(k, d, j, wj, buy_th, sell_th):
    # All inputs are broadcast to (Time, Symbol, Params) shape implicitly by Factory?
    # No, Factory runs this function for each Param set typically, or broadcasts inputs.

    # Gold Cross: prev_k < prev_d AND k > d
    gold_cross = (k > d) & (k.shift(1) < d.shift(1))

    # Death Cross: prev_k > prev_d AND k < d
    death_cross = (k < d) & (k.shift(1) > d.shift(1))

    # Trend Filter: Weekly K > Weekly D (Uptrend)
    # trend_condition = wj < 30

    # ENTRY: Gold Cross & J<BuyTh & Trend UP
    entries = gold_cross & (j < buy_th)
    exits = death_cross & (j > sell_th)

    return entries, exits


SignalFactory = vbt.IndicatorFactory(
    class_name="KDJSignal",
    short_name="signal",
    input_names=["k", "d", "j", "prev_k", "prev_d", "wk", "wd"],
    param_names=["buy_th", "sell_th"],
    output_names=["entries", "exits"],
).from_apply_func(apply_signal_logic)


def calculate_kdj(close, high, low, N=9, M1=3, M2=3):
    """
    Vectorized KDJ Calculation.
    Returns J series.
    """
    # 1. RSV
    lowest_low = low.rolling(window=N).min()
    highest_high = high.rolling(window=N).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low) * 100

    # 2. SMA for K and D?
    # Standard KDJ uses EMA-like smoothing: K = 2/3 PrevK + 1/3 RSV.
    # This is equivalent to Pandas ewm(alpha=1/3, adjust=False).

    k = rsv.ewm(alpha=1 / M1, adjust=False).mean()
    d = k.ewm(alpha=1 / M2, adjust=False).mean()
    j = 3 * k - 2 * d

    return k, d, j


def run_vectorbt_backtest():
    print("Loading data via backtesting.data_loader...")
    data_map = load_data(DATA_DIR, period_suffix="day")

    if not data_map:
        print("No data found.")
        return

    # Use centralized cleaning
    _, high_df, low_df, close_df, _ = align_and_clean_data(data_map)

    print("Calculating KDJ...")
    k, d, j = calculate_kdj(close_df, high_df, low_df)

    # Calc Weekly KDJ
    close_w = close_df.resample("W").last()
    high_w = high_df.resample("W").max()
    low_w = low_df.resample("W").min()
    k_w, d_w, j_w = calculate_kdj(close_w, high_w, low_w)

    # Align Weekly KDJ to Daily Index
    k_w = k_w.reindex(close_df.index).ffill()
    d_w = d_w.reindex(close_df.index).ffill()
    j_w = j_w.reindex(close_df.index).ffill()
    signals_is = apply_signal_logic(k, d, j, j_w, buy_th=30, sell_th=70)

    print("Running Portfolio Simulation...")
    # Portfolio.from_signals
    # init_cash=1M, fees=0.03%, slippage=0.05%
    # Note: vbt fees are fractional (0.0003), slippage is price impact

    portfolio = vbt.Portfolio.from_signals(
        close=close_df,
        entries=signals_is[0],
        exits=signals_is[1],
        init_cash=1000000,
        fees=0.0003,
        slippage=0.0005,
        freq="1D",  # frequency for annualization
    )

    print("\n" + "=" * 40)
    print("VECTORBT RESULTS (Aggregated)")
    print("=" * 40)

    # 1. Aggregate Stats (Treat all symbols as one portfolio)
    # group_by=True tells vbt to aggregate all columns
    stats = portfolio.stats(group_by=True)
    print(stats)

    print("\nDetailed Stats per symbol saved to 'vbt_results_per_symbol.csv'...")
    # metrics per symbol

    portfolio.stats().to_csv(os.path.join(REPORT_DIR, "vbt_results_per_symbol.csv"))

    print("Saving trades to 'vbt_trades.csv'...")

    portfolio.trades.records_readable.to_csv(os.path.join(REPORT_DIR, "vbt_trades.csv"))

    # Visualization & Analysis
    print("\n" + "=" * 40)
    print("COMPREHENSIVE ANALYSIS")
    print("=" * 40)

    total_returns = portfolio.total_return()

    # 1. Distribution Stats
    print(f"Mean Return:   {total_returns.mean()*100:.2f}%")
    print(f"Median Return: {total_returns.median()*100:.2f}%")
    print(f"Max Return:    {total_returns.max()*100:.2f}% ({total_returns.idxmax()})")
    print(f"Min Return:    {total_returns.min()*100:.2f}% ({total_returns.idxmin()})")

    win_rate = (total_returns > 0).mean()
    print(f"Win Rate (Stock %): {win_rate*100:.2f}%")  # % of stocks that ended positive

    print("\nGenerating comprehensive plots...")
    try:
        # 2. Return Distribution Histogram
        # Using vectorbt's plotting accessor
        hist = total_returns.vbt.histplot(
            title="Distribution of Strategy Returns across Watchlist",
            xaxis_title="Total Return",
            yaxis_title="Frequency",
        )

        hist.write_html(os.path.join(REPORT_DIR, "kdj_return_distribution.html"))
        print("  - Saved 'kdj_return_distribution.html' (Histogram)")

        # 3. Aggregate Equity Curve (The "Real" Performance)
        # Sum of value across all columns (assuming equal starting capital alloc was handled by from_signals?)
        # from_signals puts init_cash PER COLUMN by default if broadcast, or total?
        # vectorbt splits init_cash equally or applies it per column?
        # Default: init_cash is applied per column (1M per stock).
        # So summing value() gives total portfolio value.

        agg_value = portfolio.value().sum(axis=1)
        equity_curve = agg_value.vbt.plot(
            title="Aggregate Portfolio Equity (All Stocks)",
            xaxis_title="Date",
            yaxis_title="Total Equity",
        )

        equity_curve.write_html(os.path.join(REPORT_DIR, "kdj_aggregate_equity.html"))
        print("  - Saved 'kdj_aggregate_equity.html' (Aggregate Equity)")

        # 4. Best Performing Symbol (Detailed)
        best_symbol = total_returns.idxmax()
        fig_best = portfolio[best_symbol].plot()

        fig_best.write_html(
            os.path.join(REPORT_DIR, f"kdj_plot_best_{best_symbol}.html")
        )
        print(f"  - Saved 'kdj_plot_best_{best_symbol}.html' (Best Performer)")

    except Exception as e:
        print(f"Visualization failed: {e}")


if __name__ == "__main__":
    run_vectorbt_backtest()
