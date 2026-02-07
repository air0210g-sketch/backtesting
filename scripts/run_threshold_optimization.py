import vectorbt as vbt
import pandas as pd
import numpy as np
import os
import sys

# Add parent path to import data_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtesting.data_loader import load_data, align_and_clean_data

# Report Directory
REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock_data")
if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

def calculate_kdj_base(close, high, low, n=9, m=3):
    """
    Standard KDJ calculation (N=9, M=3).
    Returns K, D, J series.
    """
    lowest_low = low.rolling(window=n).min()
    highest_high = high.rolling(window=n).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
    
    k = rsv.ewm(alpha=1/m, adjust=False).mean()
    d = k.ewm(alpha=1/m, adjust=False).mean()
    j = 3 * k - 2 * d
    
    return k, d, j

# Define Signal Factory
# Inputs: k, d, j (all pre-calculated)
# Params: buy_th, sell_th
def generate_signals(k, d, j, buy_th, sell_th):
    # k, d, j are pandas objects or numpy arrays depending on vbt config
    # To be safe and easy, we assume vbt passes them as numpy arrays if generic, 
    # but we can force pandas if we needed .rolling, but here we just do elem-wise comparisons.
    # Elem-wise works on both.
    
    # Needs to handle shift for Crossover
    # k, d, j are (Time x Symbol)
    # vbt.utils.array_nb.shift_1d works on 2D? No, 1D.
    # But if we use vbt.IndicatorFactory.from_apply_func, it usually handles iteration or broadcasting.
    
    # Let's convert to DataFrame to use .shift() if it's pandas, 
    # OR better: Use vbt.IndicatorFactory.from_pandas_func if we want to work with pandas.
    # OR: Just use simple vbt broadcasting if we do this OUTSIDE the factory?
    
    # Actually, the most robust way in VBT for this specific case ("Signal Optimization"):
    # 1. Calculate inputs
    # 2. Use factory to generate entries/exits
    
    # Let's assume input is numpy array for speed.
    pass

# Simplified Approach:
# Use `vbt.IndicatorFactory.from_choice_func`? No.
# Use `vbt.IndicatorFactory.from_apply_func`
# We need `prev_k` and `prev_d` as inputs too to avoid computing shift inside!

def apply_signal_logic(k, d, j, prev_k, prev_d, wk, wd, buy_th, sell_th):
    # All inputs are broadcast to (Time, Symbol, Params) shape implicitly by Factory?
    # No, Factory runs this function for each Param set typically, or broadcasts inputs.
    
    # Gold Cross: prev_k < prev_d AND k > d
    gold_cross = (prev_k < prev_d) & (k > d)
    
    # Death Cross: prev_k > prev_d AND k < d
    death_cross = (prev_k > prev_d) & (k < d)
    
    # Trend Filter: Weekly K > Weekly D (Uptrend)
    trend_condition = wk > wd
    
    # ENTRY: Gold Cross & J<BuyTh & Trend UP
    entries = gold_cross & (j < buy_th) & trend_condition
    exits = death_cross & (j > sell_th)
    
    return entries, exits

SignalFactory = vbt.IndicatorFactory(
    class_name='KDJSignal',
    short_name='signal',
    input_names=['k', 'd', 'j', 'prev_k', 'prev_d', 'wk', 'wd'],
    param_names=['buy_th', 'sell_th'],
    output_names=['entries', 'exits']
).from_apply_func(apply_signal_logic)

def run_threshold_optimization():
    print("正在加载数据...")
    data_map = load_data(DATA_DIR, period_suffix="day")
    
    if not data_map: 
        print("未找到数据。")
        return

    # Use centralized cleaning
    _, high, low, close, _ = align_and_clean_data(data_map)
    
    # 1. Calc Base KDJ (N=9, M=3) - Standard
    # Calculate on FULL dataset to preserve trend context for OOS start
    print("正在计算基础 KDJ (N=9, M=3)...")
    k, d, j = calculate_kdj_base(close, high, low, n=9, m=3)
    
    # ---------------------------------------------------------
    # NEW: Weekly KDJ Trend Filter
    # ---------------------------------------------------------
    print("正在计算周线 KDJ (趋势过滤)...")
    # Resample to Weekly (taking last close, max high, min low)
    close_w = close.resample('W').last()
    high_w = high.resample('W').max()
    low_w = low.resample('W').min()
    
    # Calc Weekly KDJ
    k_w, d_w, j_w = calculate_kdj_base(close_w, high_w, low_w, n=9, m=3)
    
    # Broadcast Weekly KDJ back to Daily (Forward Fill)
    # Issue: Weekly resample creates Sundays, Daily data is Mon-Fri. Direct reindex misses everything.
    # Fix: Resample Weekly to Daily (D) first, then reindex.
    k_w_daily = k_w.resample('D').ffill().reindex(close.index).ffill()
    d_w_daily = d_w.resample('D').ffill().reindex(close.index).ffill()
    
    # ---------------------------------------------------------

    # 2. Split Data (IS / OOS)
    # We split Index, then slice everything.
    split_date = pd.Timestamp("2022-06-20")
    print(f"\n数据分割日期: {split_date}")
    
    is_mask = close.index < split_date
    oos_mask = close.index >= split_date
    
    print(f"样本内 (训练集): {is_mask.sum()} 天")
    print(f"样本外 (测试集): {oos_mask.sum()} 天")

    # Slice Data
    k_is = k.loc[is_mask]
    d_is = d.loc[is_mask]
    j_is = j.loc[is_mask]
    close_is = close.loc[is_mask]

    # Slice Trend Data
    wk_is = k_w_daily.loc[is_mask]
    wd_is = d_w_daily.loc[is_mask]
    
    k_oos = k.loc[oos_mask]
    d_oos = d.loc[oos_mask]
    j_oos = j.loc[oos_mask]
    close_oos = close.loc[oos_mask]

    # Slice Trend Data OOS
    wk_oos = k_w_daily.loc[oos_mask]
    wd_oos = d_w_daily.loc[oos_mask]

    # Pre-calc shifted values for crossover logic
    pk_is = k_is.shift(1)
    pd_is = d_is.shift(1)
    
    pk_oos = k_oos.shift(1)
    pd_oos = d_oos.shift(1)
    
    # 3. Define Parameter Grid
    buy_range = np.arange(10, 55, 5) # 10, 15, ..., 50
    sell_range = np.arange(60, 100, 5) # 60, 65, ..., 95
    
    print(f"\n[样本内] 正在优化阈值 ({len(buy_range)*len(sell_range)} 组参数)...")
    
    # 4. Run Signal Factory on IS Data
    signals_is = SignalFactory.run(
        k_is, d_is, j_is, pk_is, pd_is, wk_is, wd_is,
        buy_th=buy_range,
        sell_th=sell_range,
        param_product=True
    )

    # 5. Simulate Portfolio (IS)
    pf_is = vbt.Portfolio.from_signals(
        close_is,
        signals_is.entries,
        signals_is.exits,
        init_cash=1000000,
        fees=0.0003,
        slippage=0.0005,
        freq='1D'
    )
    
    # 6. Analyze IS Results
    total_returns_is = pf_is.total_return()
    mean_ret_is = total_returns_is.groupby(level=['signal_buy_th', 'signal_sell_th']).mean()
    
    best_params = mean_ret_is.idxmax() # (buy_th, sell_th)
    best_ret_is = mean_ret_is.max()
    
    print("\n" + "="*40)
    print("样本内结果 (优化)")
    print("="*40)
    print(f"最佳参数: 买入<{best_params[0]}, 卖出>{best_params[1]}")
    print(f"最佳样本内平均收益: {best_ret_is*100:.2f}%")
    
    # 7. Validation: Run Best Params on OOS Data
    print(f"\n[样本外] 正在测试最佳参数...")
    
    # Note: We pass single values (lists of length 1) to avoid product mode or broadcasting issues if not careful
    signals_oos = SignalFactory.run(
        k_oos, d_oos, j_oos, pk_oos, pd_oos, wk_oos, wd_oos,
        buy_th=[best_params[0]],
        sell_th=[best_params[1]],
        param_product=True 
    )
    
    pf_oos = vbt.Portfolio.from_signals(
        close_oos,
        signals_oos.entries,
        signals_oos.exits,
        init_cash=1000000,
        fees=0.0003,
        slippage=0.0005,
        freq='1D'
    )
    
    # 8. Final Report
    print("\n" + "="*40)
    print("样本外结果 (前向测试)")
    print("="*40)
    
    # Calculate OOS Metrics
    try:
        sys.stdout.flush()
        
        # Calculate OOS Metrics
        total_ret_oos = pf_oos.total_return()
        
        mean_ret_oos = total_ret_oos.mean()
        win_rate_oos = (total_ret_oos > 0).mean()
        
        # Sharpe & Drawdown (Daily)
        try:
            sharpe_oos = pf_oos.sharpe_ratio().mean()
            drawdown_oos = pf_oos.max_drawdown().mean()
        except:
            sharpe_oos = np.nan
            drawdown_oos = np.nan
        
        print(f"策略指标 (买入<{best_params[0]}, 卖出>{best_params[1]}) @ 未见数据:")
        print(f"平均总收益:     {mean_ret_oos*100:.2f}%")
        print(f"平均夏普比率:   {sharpe_oos:.2f}")
        print(f"平均最大回撤:   {drawdown_oos*100:.2f}%")
        print(f"胜率 (股票%):   {win_rate_oos*100:.1f}%")
        
        print("\n对比:")
        print(f"样本内 (IS) 收益:   {best_ret_is*100:.2f}%")
        print(f"样本外 (OOS) 收益:  {mean_ret_oos*100:.2f}%")
        print(f"性能衰减:          {(mean_ret_oos - best_ret_is)*100:.2f} pts")
        
        if mean_ret_oos < best_ret_is * 0.5:
            print("警告: 检测到严重过拟合！")
            
        print("\n正在生成报告文件...")
        
        # IS Heatmap (Parameter Stability)
        # Note: mean_ret_is is (buy, sell) MultiIndex Series
        heatmap_data = mean_ret_is.unstack()
        fig = heatmap_data.vbt.heatmap(
            title="样本内参数热力图 (平均收益)",
            xaxis_title="卖出阈值",
            yaxis_title="买入阈值"
        )
        fig.write_html(os.path.join(REPORT_DIR, "kdj_threshold_is_optimization.html"))
        print("已保存 'kdj_threshold_is_optimization.html'")
        
        # OOS Cumulative Returns
        oos_equity_mean = pf_oos.value().mean(axis=1)
        fig_equity = oos_equity_mean.vbt.plot(title="样本外权益曲线 (平均)")
        fig_equity.write_html(os.path.join(REPORT_DIR, "kdj_oos_equity_curve.html"))
        print("已保存 'kdj_oos_equity_curve.html'")
        
    except Exception as e:
        print(f"ERROR in Reporting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_threshold_optimization()
