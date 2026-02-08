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
    KDJ + Volume Breakout Strategy。若框架注入 k_df/d_df/j_df 则用预计算，否则内部算一次。
    """
    n, m = 9, 3
    if k_df is not None and d_df is not None and j_df is not None:
        k, d, j = k_df, d_df, j_df
    else:
        k, d, j = inds.calc_kdj(close_df, high_df, low_df, N=n, M=m)
    vol_breakout = inds.check_volume_agv_breakout(vol_df, multiple=multiple_volume_agv)

    # 2. Entry Signal
    # Condition: KDJ Gold Cross AND J < threshold_low AND Volume Breakout
    long_signal = inds.get_kdj_cross_signals(k, d, j, threshold=kdj_low, mode="long")
    entries = long_signal & vol_breakout

    # 3. Exit Signal
    # Condition: KDJ Death Cross AND J > threshold_high
    exits = (
        inds.get_kdj_cross_signals(k, d, j, threshold=kdj_high, mode="short")
        & vol_breakout
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

    # 预计算：默认周 KDJ；日 KDJ 用 ["kdj"]，加 ATR 用 ["weekly_kdj", "atr"]
    runner.set_precompute_indicators("default")

    # Load Data
    if not runner.load_data():
        exit(1)

    # Define Parameter Grid
    # Focus on thresholds and volume settings
    param_grid = {
        "kdj_low": np.arange(25, 40, 3).astype(int).tolist(),
        "kdj_high": np.arange(50, 90, 5).astype(int).tolist(),
        "multiple_volume_agv": np.arange(1.25, 1.51, 0.05).astype(float).tolist(),
        "sl_stop": np.arange(0.15, 0.31, 0.05).astype(float).tolist(),
        "sl_trail": [0.2],
        # "sl_trail": np.arange(0.15, 0.31, 0.05).astype(float).tolist(),
    }

    # Run Full Cycle (Opt -> Valid -> Report -> Plot)
    # 组合过多时可选：n_jobs=4 多进程并行批次数；sample_max=200 随机抽样 200 组信号参数；chunk_size_combos=100 每批更多组合（省批次数、略增内存）
    print("Starting Backtest Cycle with Accumulation...")
    runner.run_full_cycle(
        my_strategy_logic,
        param_grid,
        split_ratio=0.7,
        accumulate=True,
        size=0.10,
        size_type="value",
        # n_jobs=4,           # 并行批次数，可显著缩短耗时
        # chunk_size_combos=100,
        # sample_max=200,     # 仅随机抽样 200 组信号参数做优化
    )
    print("#" * 100)
    print("#" * 100)
