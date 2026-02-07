
import vectorbt as vbt
import pandas as pd
import numpy as np
import talib
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

def get_pattern_functions():
    """Returns a dictionary of all pattern recognition functions in TA-Lib."""
    all_groups = talib.get_function_groups()
    pattern_names = all_groups.get('Pattern Recognition', [])
    patterns = {}
    for name in pattern_names:
        patterns[name] = getattr(talib, name)
    return patterns

def run_talib_analysis():
    print("Loading data...")
    data_map = load_data(DATA_DIR, period_suffix="day")
    
    if not data_map:
        print("No data found.")
        return

    print(f"Aligning data for {len(data_map)} symbols...")
    close_df = pd.DataFrame({sym: df['close'] for sym, df in data_map.items()})
    open_df = pd.DataFrame({sym: df['open'] for sym, df in data_map.items()})
    high_df = pd.DataFrame({sym: df['high'] for sym, df in data_map.items()})
    low_df = pd.DataFrame({sym: df['low'] for sym, df in data_map.items()})
    vol_df = pd.DataFrame({sym: df['volume'] for sym, df in data_map.items()})

    pattern_funcs = get_pattern_functions()
    print(f"Found {len(pattern_funcs)} TA-Lib pattern functions.")
    
    results = []

    # Iterate through each pattern
    for name, func in pattern_funcs.items():
        print(f"Testing pattern: {name}...")
        try:
            # TA-Lib functions typically take Open, High, Low, Close
            # Some need Volume? Most Pattern Recog just need OHLC.
            # We need to apply this function to each column of our dataframes.
            # VectorBT doesn't natively vectorize TA-Lib (which is C-based 1D arrays).
            # We can use vbt.IndicatorFactory or just pandas apply.
            # Since we have 90 stocks, pandas apply is acceptable speed.
            
            # Helper to apply to one series
            def apply_pattern(o, h, l, c):
                # Inputs are Series, convert to values for ta-lib if needed, 
                # but ta-lib handles numpy arrays.
                return func(o.values, h.values, l.values, c.values)
            
            # Apply to all columns
            # This is a bit slow. Optimization: Use numpy apply_along_axis?
            # Or loop columns.
            
            # Let's loop columns for clarity and safety with TA-Lib's strict array requirements
            pattern_signals = pd.DataFrame(index=close_df.index, columns=close_df.columns)
            
            for col in close_df.columns:
                # Prepare inputs
                # TA-Lib requires double (float64).
                s_o = open_df[col].values.astype(float)
                s_h = high_df[col].values.astype(float)
                s_l = low_df[col].values.astype(float)
                s_c = close_df[col].values.astype(float)
                
                # Execute
                res = func(s_o, s_h, s_l, s_c)
                pattern_signals[col] = res
                
            # Pattern Logic:
            # 100 = Bullish (Buy)
            # -100 = Bearish (Sell)
            # 0 = No Pattern
            
            entries = pattern_signals == 100
            exits = pattern_signals == -100
            
            # If a pattern only detects entries (never returns -100), we need an exit strategy?
            # For pure testing of the pattern's predictive power, we might use a fixed hold or just see if the "Sell" signal works.
            # If entries is empty, skip
            if not entries.any().any():
                continue
                
            # If exits is empty, maybe we use a trailing stop or just Fixed Hold?
            # User request: "Buy > 0, Sell < 0". Strict interpretation.
            if not exits.any().any():
                # If no sell signals, we can't close trades based on pattern.
                # Let's add a generic STOP LOSS or TIME EXIT to make it viable?
                # Or just strictly follow user logic (might result in open positions).
                pass
                
            # Simulate
            pf = vbt.Portfolio.from_signals(
                close=close_df,
                entries=entries,
                exits=exits,
                init_cash=1000000,
                fees=0.0003,
                slippage=0.0005,
                freq='1D'
            )
            
            total_ret = pf.total_return().mean()
            win_rate = (pf.total_return() > 0).mean()
            trades = pf.trades.count().sum()
            
            results.append({
                "Pattern": name,
                "Mean Return": total_ret,
                "Win Rate": win_rate,
                "Trade Count": trades
            })
            
        except Exception as e:
            print(f"Error processing {name}: {e}")

    # Summary
    if not results:
        print("No patterns generated trades.")
        return

    results_df = pd.DataFrame(results).sort_values(by="Mean Return", ascending=False)
    
    # Save Summary
    output_path = os.path.join(REPORT_DIR, "talib_patterns_summary.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved pattern summary to '{output_path}'")
    
    print("\nTop 10 Patterns:")
    print(results_df.head(10))
    
    # Analyze Best Pattern
    best_pattern = results_df.iloc[0]["Pattern"]
    print(f"\nGenerating detailed report for Best Pattern: {best_pattern}...")
    
    # Re-run best pattern to get portfolio object
    func = get_pattern_functions()[best_pattern]
    best_signals = pd.DataFrame(index=close_df.index, columns=close_df.columns)
    for col in close_df.columns:
        s_o = open_df[col].values.astype(float)
        s_h = high_df[col].values.astype(float)
        s_l = low_df[col].values.astype(float)
        s_c = close_df[col].values.astype(float)
        best_signals[col] = func(s_o, s_h, s_l, s_c)
        
    entries = best_signals == 100
    exits = best_signals == -100
    
    pf_best = vbt.Portfolio.from_signals(
        close=close_df,
        entries=entries,
        exits=exits,
        init_cash=1000000,
        fees=0.0003,
        slippage=0.0005,
        freq='1D'
    )
    
    agg_value = pf_best.value().sum(axis=1)
    fig = agg_value.vbt.plot(
        title=f"Aggregate Equity: Best Pattern ({best_pattern})",
        xaxis_title="Date",
        yaxis_title="Equity"
    )
    fig.write_html(os.path.join(REPORT_DIR, "talib_best_pattern.html"))
    print("Saved 'talib_best_pattern.html'")

if __name__ == "__main__":
    run_talib_analysis()
