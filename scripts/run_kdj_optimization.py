import vectorbt as vbt
import pandas as pd
import numpy as np
import os
import sys

# Add current path to import data_loader
# Add parent path to import data_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtesting.data_loader import load_data

# Report Directory
REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock_data")
if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

def calculate_kdj_custom(close, high, low, n, m):
    """
    Custom KDJ function.
    N: Rolling window for RSV
    M: Smoothing factor for K and D (K uses M, D uses M)
    """
    # Convert to DataFrame to use pandas functions easily
    # Inputs from vbt are typically 2D numpy arrays (Time x Symbols)
    close = pd.DataFrame(close)
    high = pd.DataFrame(high)
    low = pd.DataFrame(low)
    
    # 1. RSV
    lowest_low = low.rolling(window=n).min()
    highest_high = high.rolling(window=n).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
    
    # 2. SMA/EMA
    k = rsv.ewm(alpha=1/m, adjust=False).mean()
    d = k.ewm(alpha=1/m, adjust=False).mean()
    j = 3 * k - 2 * d
    
    return k, d, j

# Define Indicator Factory
KDJIndicator = vbt.IndicatorFactory(
    class_name='KDJ',
    short_name='kdj',
    input_names=['close', 'high', 'low'],
    param_names=['n', 'm'],
    output_names=['k', 'd', 'j']
).from_apply_func(calculate_kdj_custom)

def run_optimization():
    print("Loading data...")
    data_map = load_data(DATA_DIR, period_suffix="day")
    
    if not data_map:
        return

    print(f"Aligning data for {len(data_map)} symbols...")
    close_dict = {sym: df['close'] for sym, df in data_map.items()}
    high_dict = {sym: df['high'] for sym, df in data_map.items()}
    low_dict = {sym: df['low'] for sym, df in data_map.items()}
    
    close_df = pd.DataFrame(close_dict)
    high_df = pd.DataFrame(high_dict)
    low_df = pd.DataFrame(low_dict)
    
    # Parameter Grid
    # N: 9 to 30, step 3
    # M: 3 to 5
    n_range = np.arange(9, 31, 3) 
    m_range = [3, 4, 5]
    
    print(f"Running Optimization across {len(n_range)*len(m_range)} parameter combinations...")
    print(f"Total Backtests: {len(n_range)*len(m_range)} Params * {len(data_map)} Stocks = {len(n_range)*len(m_range)*len(data_map)}")
    
    # Run Indicator
    # param_product=True means run all combinations of n and m
    kdj = KDJIndicator.run(
        close_df, high_df, low_df,
        n=n_range,
        m=m_range,
        param_product=True
    )
    
    # Logic
    prev_k = kdj.k.shift(1)
    prev_d = kdj.d.shift(1)
    
    entries = (prev_k < prev_d) & (kdj.k > kdj.d) & (kdj.j < 30)
    exits = (prev_k > prev_d) & (kdj.k < kdj.d) & (kdj.j > 90)
    
    print("Simulating Portfolio...")
    portfolio = vbt.Portfolio.from_signals(
        close_df,
        entries,
        exits,
        init_cash=1000000,
        fees=0.0003,
        slippage=0.0005,
        freq='1D'
    )
    
    print("\n" + "="*40)
    print("OPTIMIZATION RESULTS")
    print("="*40)
    
    # Analyze Best Parameters
    # We want to know which (n, m) gives the best "Total Return" (averaged across all stocks, or median)
    
    total_return = portfolio.total_return()
    # total_return structure: Index=(kdj_n, kdj_m), Columns=Symbol
    
    # Mean return across all stocks for each parameter set
    mean_return_by_param = total_return.groupby(level=['kdj_n', 'kdj_m']).mean()
    
    best_params = mean_return_by_param.idxmax()
    best_return = mean_return_by_param.max()
    
    print(f"Best Parameters (Avg Return): N={best_params[0]}, M={best_params[1]}")
    print(f"Best Avg Return: {best_return*100:.2f}%")
    
    print("\nTop 5 Parameter Sets:")
    print(mean_return_by_param.sort_values(ascending=False).head(5) * 100)
    
    # Visualization: Heatmap
    # Unstack to get Matrix: Index=N, Columns=M
    print("\nGenerating Heatmap...")
    try:
        heatmap_data = mean_return_by_param.unstack()
        fig = heatmap_data.vbt.heatmap(
            title="KDJ Optimization: Mean Return by (N, M)",
            xaxis_title="Smoothing (M)",
            yaxis_title="Window (N)"
        )

        fig.write_html(os.path.join(REPORT_DIR, "kdj_optimization_heatmap.html"))
        print("Heatmap saved to 'kdj_optimization_heatmap.html'")
        
        # Also detailed plot for the Best Parameter Set
        # Re-run stats for best
        # Actually we can slice the portfolio object!
        best_portfolio = portfolio.xs(best_params, level=['kdj_n', 'kdj_m'])
        
        # Generate aggregate equity for the best param
        agg_value = best_portfolio.value().sum(axis=1)
        fig_equity = agg_value.vbt.plot(title=f"Aggregate Equity (Best Params: N={best_params[0]}, M={best_params[1]})")

        fig_equity.write_html(os.path.join(REPORT_DIR, "kdj_best_param_equity.html"))
        print("Best Param Equity saved to 'kdj_best_param_equity.html'")
        
    except Exception as e:
        print(f"Visualization failed: {e}")

if __name__ == "__main__":
    run_optimization()
