
import vectorbt as vbt
import pandas as pd
import numpy as np
import os
import sys

# Add parent path to import data_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtesting.data_loader import load_data

# Report Directory
REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock_data")
if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

def calculate_kdj(close, high, low, N=9, M=4):
    """
    KDJ Calculation with custom M.
    """
    # 1. RSV
    lowest_low = low.rolling(window=N).min()
    highest_high = high.rolling(window=N).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
    
    # 2. SMA for K and D
    k = rsv.ewm(alpha=1/M, adjust=False).mean()
    d = k.ewm(alpha=1/M, adjust=False).mean()
    j = 3 * k - 2 * d
    
    return k, d, j

def run_final_analysis():
    print("Loading data...")
    data_map = load_data(DATA_DIR, period_suffix="day")
    
    if not data_map:
        print("No data found.")
        return

    print(f"Aligning data for {len(data_map)} symbols...")
    close_df = pd.DataFrame({sym: df['close'] for sym, df in data_map.items()})
    high_df = pd.DataFrame({sym: df['high'] for sym, df in data_map.items()})
    low_df = pd.DataFrame({sym: df['low'] for sym, df in data_map.items()})
    
    # BEST PARAMETERS
    N = 9
    M = 4
    BUY_TH = 15
    SELL_TH = 92
    
    print(f"Running Final Backtest with: N={N}, M={M}, Buy<{BUY_TH}, Sell>{SELL_TH}")
    
    print("Calculating KDJ...")
    k, d, j = calculate_kdj(close_df, high_df, low_df, N=N, M=M)
    
    # Logic: Gold Cross & J < Buy_TH
    prev_k = k.shift(1)
    prev_d = d.shift(1)
    
    gold_cross = (prev_k < prev_d) & (k > d)
    death_cross = (prev_k > prev_d) & (k < d)
    
    entries = gold_cross & (j < BUY_TH)
    exits = death_cross & (j > SELL_TH)
    
    print("Simulating Portfolio...")
    portfolio = vbt.Portfolio.from_signals(
        close=close_df,
        entries=entries,
        exits=exits,
        init_cash=1000000,
        fees=0.0003,
        slippage=0.0005,
        freq='1D'
    )
    
    total_returns = portfolio.total_return()
    
    print("\n" + "="*40)
    print("FINAL ANALYSIS RESULTS")
    print("="*40)
    
    # 1. Distribution Stats
    mean_ret = total_returns.mean()
    median_ret = total_returns.median()
    max_ret = total_returns.max()
    min_ret = total_returns.min()
    win_rate = (total_returns > 0).mean()
    
    print(f"Mean Return:   {mean_ret*100:.2f}%")
    print(f"Median Return: {median_ret*100:.2f}%")
    print(f"Max Return:    {max_ret*100:.2f}% ({total_returns.idxmax()})")
    print(f"Min Return:    {min_ret*100:.2f}% ({total_returns.idxmin()})")
    print(f"Win Rate:      {win_rate*100:.2f}%")
    
    # 2. Histogram
    hist_fig = total_returns.vbt.histplot(
        title=f"Return Distribution (N={N}, M={M}, Buy<{BUY_TH}, Sell>{SELL_TH})",
        xaxis_title="Total Return",
        yaxis_title="Count"
    )
    hist_fig.write_html(os.path.join(REPORT_DIR, "final_return_hist.html"))
    print("Saved 'final_return_hist.html'")
    
    # 3. Top 5 Stocks Analysis
    top_5 = total_returns.sort_values(ascending=False).head(5)
    print("\nTop 5 Performing Stocks:")
    print(top_5 * 100)
    
    # Generate Plots for Top 5
    for sym in top_5.index:
        print(f"Generating detailed plot for {sym}...")
        try:
            # Create a subplot with Price/Signals and Equity
            pf_sym = portfolio[sym]
            
            # Simple combined plot via vbt
            # VBT plot() usually shows signals on price + equity
            fig = pf_sym.plot()
            fig.write_html(os.path.join(REPORT_DIR, f"final_plot_{sym}.html"))
        except Exception as e:
            print(f"Error plotting {sym}: {e}")
            
    # Save Top 5 Metrics to CSV
    # We want more than just return: Trades, Win Rate, Max DD, etc.
    top_5_stats = portfolio[top_5.index].stats()
    top_5_stats.to_csv(os.path.join(REPORT_DIR, "final_top5_metrics.csv"))
    print("Saved 'final_top5_metrics.csv'")

if __name__ == "__main__":
    run_final_analysis()
