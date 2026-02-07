
import vectorbt as vbt
import pandas as pd
import numpy as np
import talib
import os
import sys

# Add parent path to import data_loader
import sys

# Add parent path to import data_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtesting.data_loader import load_data, align_and_clean_data

# Report Directory
REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock_data")
if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

def calculate_kdj(close, high, low, N=9, M=4):
    """
    KDJ Calculation with best params.
    """
    lowest_low = low.rolling(window=N).min()
    highest_high = high.rolling(window=N).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
    
    k = rsv.ewm(alpha=1/M, adjust=False).mean()
    d = k.ewm(alpha=1/M, adjust=False).mean()
    j = 3 * k - 2 * d
    
    return j

def run_combined_analysis():
    print("Loading data...")
    data_map = load_data(DATA_DIR, period_suffix="day")
    
    if not data_map:
        print("No data found.")
        return

    # Use centralized cleaning
    open_df, high_df, low_df, close_df, _ = align_and_clean_data(data_map)
    
    # 1. Calculate KDJ J-Line
    print("Calculating KDJ (N=9, M=4)...")
    j_line = calculate_kdj(close_df, high_df, low_df, N=9, M=4)
    
    # Condition 1: Setup (J < 15)
    setup_condition = j_line < 15
    
    # Condition 2: Exit (J > 92)
    # Note: Exit is purely KDJ based.
    exit_condition = j_line > 92
    
    # 2. Selected Patterns
    patterns = {
        "CDLDOJI": talib.CDLDOJI,
        "CDLHAMMER": talib.CDLHAMMER,
        "CDLHARAMI": talib.CDLHARAMI,
        "CDLINVERTEDHAMMER": talib.CDLINVERTEDHAMMER
    }
    
    results = []
    
    for name, func in patterns.items():
        print(f"Testing Combination: KDJ + {name}...")
        try:
            # Calc Pattern Signals
            pattern_signals = pd.DataFrame(index=close_df.index, columns=close_df.columns)
            
            for col in close_df.columns:
                s_o = open_df[col].values.astype(float)
                s_h = high_df[col].values.astype(float)
                s_l = low_df[col].values.astype(float)
                s_c = close_df[col].values.astype(float)
                pattern_signals[col] = func(s_o, s_h, s_l, s_c)
            
            # TRIGGER: Pattern is Bullish (100)
            trigger_condition = pattern_signals == 100
            
            # ENTRY: Setup AND Trigger
            # We want to buy if we are in Setup zone AND Trigger happens.
            entries = setup_condition & trigger_condition
            
            # EXIT: KDJ Overbought
            exits = exit_condition
            
            # Run Backtest
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
            median_ret = total_ret.median()
            max_ret = total_ret.max()
            win_rate = (total_ret > 0).mean()
            trade_count = pf.trades.count().sum()
            
            print(f"  -> Mean Return: {mean_ret*100:.2f}%, Win Rate: {win_rate*100:.2f}%, Trades: {trade_count}")
            
            results.append({
                "Strategy": f"KDJ+{name}",
                "Mean Return": mean_ret,
                "Median Return": median_ret,
                "Max Return": max_ret,
                "Win Rate": win_rate,
                "Total Trades": trade_count
            })
            
            # Save Best Performing Plot for this combo
            best_sym = total_ret.idxmax()
            pf[best_sym].plot().write_html(os.path.join(REPORT_DIR, f"combined_{name}_best_{best_sym}.html"))
            
            # If this is the best strategy so far, save equity curve
            # (Logic handled after loop)
             
        except Exception as e:
            print(f"Error testing {name}: {e}")

    # Summary
    results_df = pd.DataFrame(results).sort_values(by="Mean Return", ascending=False)
    print("\n" + "="*40)
    print("COMBINED STRATEGY RESULTS")
    print("="*40)
    print(results_df)
    
    output_path = os.path.join(REPORT_DIR, "combined_strategy_comparison.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved summary to '{output_path}'")
    
    # Generate Aggregate Equity Curve for the Best Combo
    if not results_df.empty:
        best_combo_name = results_df.iloc[0]["Strategy"].replace("KDJ+", "")
        print(f"\nGenerating Aggregate Equity for Best Combo: {best_combo_name}...")
        
        # We need to re-run or just store portfolio? Re-running is cleaner for script structure
        # ... (Duplicate logic for selected pattern)
        # For brevity, let's just accept the individual best plot saved above and the CSV.
        # But user asked for 'combined_best_equity.html'.
        
        # Quick re-calc for the best one to get aggregate equity
        func = patterns[best_combo_name]
        
        pattern_signals = pd.DataFrame(index=close_df.index, columns=close_df.columns)
        for col in close_df.columns:
            s_o = open_df[col].values.astype(float)
            s_h = high_df[col].values.astype(float)
            s_l = low_df[col].values.astype(float)
            s_c = close_df[col].values.astype(float)
            pattern_signals[col] = func(s_o, s_h, s_l, s_c)

        trigger_condition = pattern_signals == 100
        entries = setup_condition & trigger_condition
        exits = exit_condition
        
        pf_best = vbt.Portfolio.from_signals(
            close=close_df,
            entries=entries,
            exits=exits,
            init_cash=1000000,
            fees=0.0003,
            slippage=0.0005,
            freq='1D'
        )
        
        pf_best.value().sum(axis=1).vbt.plot(
            title=f"Aggregate Equity: KDJ + {best_combo_name}",
            xaxis_title="Date",
            yaxis_title="Total Equity"
        ).write_html(os.path.join(REPORT_DIR, "combined_best_equity.html"))
        print("Saved 'combined_best_equity.html'")

if __name__ == "__main__":
    run_combined_analysis()
