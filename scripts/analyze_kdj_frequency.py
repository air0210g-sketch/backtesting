
import vectorbt as vbt
import pandas as pd
import numpy as np
import os
import sys
import plotly.express as px

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
    KDJ Calculation (J only).
    """
    lowest_low = low.rolling(window=N).min()
    highest_high = high.rolling(window=N).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
    
    k = rsv.ewm(alpha=1/M, adjust=False).mean()
    d = k.ewm(alpha=1/M, adjust=False).mean()
    j = 3 * k - 2 * d
    
    return j

def run_frequency_analysis():
    print("Loading data...")
    data_map = load_data(DATA_DIR, period_suffix="day")
    
    if not data_map:
        print("No data found.")
        return

    print("Aligning data...")
    close_df = pd.DataFrame({sym: df['close'] for sym, df in data_map.items()})
    high_df = pd.DataFrame({sym: df['high'] for sym, df in data_map.items()})
    low_df = pd.DataFrame({sym: df['low'] for sym, df in data_map.items()})
    
    # 1. Calculate Base J-Line (N=9, M=4)
    print("Calculating KDJ (N=9, M=4)...")
    j_line =  (close_df, high_df, low_df, N=9, M=4)
    
    # 2. Define Parameter Grid
    buy_range = np.arange(0, 41, 5)   # 0, 5, ..., 40
    sell_range = np.arange(60, 101, 5) # 60, 65, ..., 100
    
    results = []
    
    print(f"Testing {len(buy_range) * len(sell_range)} combinations...")
    
    # Use VectorBT broadcasting for speed?
    # vbt.Portfolio.from_signals can accept 2D/3D signals but it might be complex to align.
    # Simple loop is clear enough for ~81 combos.
    
    for b_th in buy_range:
        for s_th in sell_range:
            # Generate Signals
            # Logic: Buy if J < b_th, Sell if J > s_th
            # Note: This is "Value Based" entry.
            # Usually KDJ uses CrossOver. But previous experiment used Value check (J < X).
            # Wait, verify_data used: entries = (gold_cross) & (j < 30).
            # Our previous threshold optimization used: entries = (gold_cross) & (j < buy_th).
            # Let's stick to the CROSS logic + Threshold filter, as that's the "Strategy".
            
            # Recalculate K,D needed for Cross?
            # Actually we just need J for filter, but we need K,D for Cross.
            # Let's re-calc full KDJ inside or pass them.
            # To be consistent with "Threshold Optimization", we need Cross Logic.
            
            # Let's do full calc once.
            pass

    # Re-doing calc outside loop
    lowest_low = low_df.rolling(window=9).min()
    highest_high = high_df.rolling(window=9).max()
    rsv = (close_df - lowest_low) / (highest_high - lowest_low) * 100
    k = rsv.ewm(alpha=1/4, adjust=False).mean()
    d = k.ewm(alpha=1/4, adjust=False).mean()
    j = 3 * k - 2 * d
    
    prev_k = k.shift(1)
    prev_d = d.shift(1)
    gold_cross = (prev_k < prev_d) & (k > d)
    death_cross = (prev_k > prev_d) & (k < d)
    
    for b_th in buy_range:
        for s_th in sell_range:
            entries = gold_cross & (j < b_th)
            exits = death_cross & (j > s_th)
            
            # Simulation
            pf = vbt.Portfolio.from_signals(
                close=close_df,
                entries=entries,
                exits=exits,
                init_cash=1000000,
                fees=0.0003,
                slippage=0.0005,
                freq='1D'
            )
            
            total_ret = pf.total_return()
            
            mean_ret = total_ret.mean()
            win_rate = (total_ret > 0).mean()
            total_trades = pf.trades.count().sum()
            
            results.append({
                "Buy_Th": b_th,
                "Sell_Th": s_th,
                "Mean_Return": mean_ret,
                "Win_Rate": win_rate,
                "Total_Trades": total_trades
            })
            
    # Analysis
    df = pd.DataFrame(results)
    
    print("\n" + "="*40)
    print("FREQUENCY ANALYSIS RESULTS")
    print("="*40)
    
    # Sort by Mean Return
    print("Top 10 by Return:")
    print(df.sort_values("Mean_Return", ascending=False).head(10))
    
    # Save CSV
    csv_path = os.path.join(REPORT_DIR, "kdj_frequency_analysis.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved data to '{csv_path}'")
    
    # Scatter Plot
    # X=Trades, Y=Return, Size=WinRate, Color=WinRate
    print("Generating Scatter Plot...")
    
    fig = px.scatter(
        df,
        x="Total_Trades",
        y="Mean_Return",
        color="Win_Rate",
        size="Win_Rate", # larger dots for higher win rate
        hover_data=["Buy_Th", "Sell_Th"],
        title="KDJ Strategy: Trade Frequency vs Mean Return (N=9, M=4)",
        labels={
            "Total_Trades": "Total Trades (3 Years, 90 Stocks)",
            "Mean_Return": "Mean Return",
            "Win_Rate": "Win Rate"
        }
    )
    
    # Add a horizontal line for 0 return
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    html_path = os.path.join(REPORT_DIR, "kdj_frequency_scatter.html")
    fig.write_html(html_path)
    print(f"Saved plot to '{html_path}'")

if __name__ == "__main__":
    run_frequency_analysis()
