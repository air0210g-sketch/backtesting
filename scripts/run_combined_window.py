
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

def calculate_kdj(close, high, low, N=9, M=4):
    """
    KDJ Calculation.
    """
    lowest_low = low.rolling(window=N).min()
    highest_high = high.rolling(window=N).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
    
    k = rsv.ewm(alpha=1/M, adjust=False).mean()
    d = k.ewm(alpha=1/M, adjust=False).mean()
    j = 3 * k - 2 * d
    
    return j

def run_combined_window_analysis():
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
    
    # 1. Calculate KDJ J-Line
    print("Calculating KDJ (N=9, M=4)...")
    j_line = calculate_kdj(close_df, high_df, low_df, N=9, M=4)
    
    # Condition 1: Setup (J < 15)
    # This is our momentary trigger for the KDJ side.
    # Actually, the user said: "If pattern appears... within 3 days KDJ signal appears".
    # Or "Pattern valid for 3 days".
    # Implementation:
    # We broaden the Pattern signal. If Pattern appeared today OR yesterday OR day before...
    # Then if J < 15 today, we buy.
    
    kdj_signal = j_line < 15
    exit_signal = j_line > 92
    
    # 2. Selected Patterns
    patterns = {
        "CDLDOJI": talib.CDLDOJI,
        "CDLHAMMER": talib.CDLHAMMER,
        "CDLHARAMI": talib.CDLHARAMI,
        "CDLINVERTEDHAMMER": talib.CDLINVERTEDHAMMER
    }
    
    results = []
    
    # Window settings
    WINDOW_SIZE = 4 # Today + 3 previous days
    
    for name, func in patterns.items():
        print(f"Testing Combination (Window={WINDOW_SIZE-1} days): KDJ + {name}...")
        try:
            # Calc Pattern Signals
            pattern_raw = pd.DataFrame(index=close_df.index, columns=close_df.columns)
            
            for col in close_df.columns:
                s_o = open_df[col].values.astype(float)
                s_h = high_df[col].values.astype(float)
                s_l = low_df[col].values.astype(float)
                s_c = close_df[col].values.astype(float)
                pattern_raw[col] = func(s_o, s_h, s_l, s_c)
            
            # TRIGGER BROADENING
            # Identify where pattern is Bullish (100)
            is_bullish = (pattern_raw == 100).astype(int)
            
            # Expand validity window
            # using rolling max: if 1 was present in last 4 days, result is 1
            pattern_valid = is_bullish.rolling(window=WINDOW_SIZE).max() > 0
            
            # ENTRY: J < 15 AND Pattern Valid (in last 3 days)
            entries = kdj_signal & pattern_valid
            
            # EXIT: KDJ Overbought
            exits = exit_signal
            
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
                "Strategy": f"KDJ+{name}(W=3)",
                "Mean Return": mean_ret,
                "Median Return": median_ret,
                "Max Return": max_ret,
                "Win Rate": win_rate,
                "Total Trades": trade_count
            })
            
             
        except Exception as e:
            print(f"Error testing {name}: {e}")

    # Summary
    results_df = pd.DataFrame(results).sort_values(by="Mean Return", ascending=False)
    print("\n" + "="*40)
    print("COMBINED STRATEGY (WINDOW) RESULTS")
    print("="*40)
    print(results_df)
    
    output_path = os.path.join(REPORT_DIR, "combined_window_comparison.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved summary to '{output_path}'")
    
    # Save best equity to html
    if not results_df.empty:
        best_strat = results_df.iloc[0]["Strategy"]
        pattern_name = best_strat.replace("KDJ+", "").replace("(W=3)", "")
        
        print(f"\nGenerating Equity Curve for {best_strat}...")
        
        # Recalc for plotting
        func = patterns[pattern_name]
        pattern_raw = pd.DataFrame(index=close_df.index, columns=close_df.columns)
        for col in close_df.columns:
            s_o = open_df[col].values.astype(float)
            s_h = high_df[col].values.astype(float)
            s_l = low_df[col].values.astype(float)
            s_c = close_df[col].values.astype(float)
            pattern_raw[col] = func(s_o, s_h, s_l, s_c)

        is_bullish = (pattern_raw == 100).astype(int)
        pattern_valid = is_bullish.rolling(window=WINDOW_SIZE).max() > 0
        entries = kdj_signal & pattern_valid
        exits = exit_signal
        
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
            title=f"Aggregate Equity: {best_strat}",
            xaxis_title="Date",
            yaxis_title="Total Equity"
        ).write_html(os.path.join(REPORT_DIR, "combined_window_best_equity.html"))

    
if __name__ == "__main__":
    run_combined_window_analysis()
