import os
import sys
import pandas as pd
import numpy as np

# Add parent path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from backtesting.framework import BacktestRunner
    import backtesting.indicators as inds
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from backtesting.framework import BacktestRunner
    import backtesting.indicators as inds

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock_data"
)
REPORT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"
)


# ----------------------------------------------------
# 1. Define Strategy Logic (Pure Function)
# ----------------------------------------------------
def my_strategy_logic(
    open_df,
    high_df,
    low_df,
    close_df,
    vol_df,
    k_df=None,
    d_df=None,
    j_df=None,
    kdj_low=40,
    kdj_high=80,
    multiple_volume_agv=1,
    **kwargs,
):
    """
    KDJ + Volume Breakout Strategy.
    若框架注入 k_df/d_df/j_df（预计算），则直接用，否则在内部算一次（n=9, m=3）。
    """
    # 1. 指标：预计算注入则用，否则算一次
    if k_df is not None and d_df is not None and j_df is not None:
        k, d, j = k_df, d_df, j_df
    else:
        n, m = 9, 3
        k, d, j = inds.calc_weekly_kdj(open_df, high_df, low_df, close_df, N=n, M=m)
    vol_breakout = inds.check_volume_agv_breakout(vol_df, multiple=multiple_volume_agv)

    # 2. Entry Signal
    # Condition: KDJ Gold Cross AND J < threshold_low AND Volume Breakout
    long_signal = inds.get_kdj_cross_signals(k, d, j, threshold=kdj_low, mode="long")
    entries = long_signal  # & vol_breakout

    # 3. Exit Signal
    # Condition: KDJ Death Cross AND J > threshold_high
    exits = (
        inds.get_kdj_cross_signals(k, d, j, threshold=kdj_high, mode="short")
        # & vol_breakout
    )

    return entries, exits


# ----------------------------------------------------
# 2. Run Framework (Config & Execute)
# ----------------------------------------------------
if __name__ == "__main__":
    print("#" * 100)
    print("#" * 100)
    # Initialize Runner
    runner = BacktestRunner(DATA_DIR, REPORT_DIR)

    # 预计算：默认周 KDJ；也可 set_precompute_indicators(["weekly_kdj", "atr"])
    runner.set_precompute_indicators("default")

    # Load Data（use_cache=True 时对齐结果缓存为 .parquet，二次启动直接读缓存）
    if not runner.load_data():
        exit(1)

    # Filter for 2015.HK specifically
    target_symbol = "700.HK"
    if target_symbol in runner.close.columns:
        print(f"Filtering data for {target_symbol}...")
        runner.close = runner.close[[target_symbol]]
        runner.open = runner.open[[target_symbol]]
        runner.high = runner.high[[target_symbol]]
        runner.low = runner.low[[target_symbol]]
        runner.volume = runner.volume[[target_symbol]]
        runner.mask = runner.mask[[target_symbol]]
    else:
        print(f"Warning: {target_symbol} not found in data. Running on all symbols.")

    # Define Parameter Grid
    # Focus on thresholds and volume settings
    param_grid = {
        "kdj_low": [31],
        "kdj_high": [84],
        "multiple_volume_agv": [1.3],
        "sl_stop": [0.20],
        "sl_trail": [0.20],
    }
    # Run Full Cycle (Opt -> Valid -> Report -> Plot)
    print("Starting Backtest Cycle...")
    runner.run_full_cycle(
        my_strategy_logic,
        param_grid,
        split_ratio=0.7,
        accumulate=True,  # 开启多次入场
        size=0.1,  # 每次入场仓位
        size_type="value",  # 按价值百分比
    )
    print("#" * 100)
    print("#" * 100)
