
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

def debug_kdj(symbol, n=9, m=3, buy_th=15, sell_th=90):
    print(f"Debugging KDJ for {symbol} (N={n}, M={m}, Buy<{buy_th}, Sell>{sell_th})...")
    
    data_map = load_data(DATA_DIR, period_suffix="day", symbols=[symbol])
    if not data_map or symbol not in data_map:
        # Fallback if load_data doesn't support list filtering correctly or no data
        data_map = load_data(DATA_DIR)
        if symbol not in data_map:
            print(f"Symbol {symbol} not found.")
            return

    df = data_map[symbol].copy()
    
    # Calculate KDJ manually and expose every step
    low_min = df['low'].rolling(n).min()
    high_max = df['high'].rolling(n).max()
    
    # 1. RSV
    df['lowest_low'] = low_min
    df['highest_high'] = high_max
    df['rsv'] = (df['close'] - low_min) / (high_max - low_min) * 100
    
    # 2. K, D, J
    # Use pandas ewm
    df['k'] = df['rsv'].ewm(alpha=1/m, adjust=False).mean()
    df['d'] = df['k'].ewm(alpha=1/m, adjust=False).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']
    
    # 3. Cross Logic
    df['prev_k'] = df['k'].shift(1)
    df['prev_d'] = df['d'].shift(1)
    
    df['gold_cross'] = (df['prev_k'] < df['prev_d']) & (df['k'] > df['d'])
    df['death_cross'] = (df['prev_k'] > df['prev_d']) & (df['k'] < df['d'])
    
    # 4. Signals
    df['is_buy'] = df['gold_cross'] & (df['j'] < buy_th)
    df['is_sell'] = df['death_cross'] & (df['j'] > sell_th)
    
    # Output to CSV for manual inspection
    columns = ['open', 'high', 'low', 'close', 'rsv', 'k', 'd', 'j', 'prev_k', 'prev_d', 'gold_cross', 'death_cross', 'is_buy', 'is_sell']
    
    output_file = os.path.join(REPORT_DIR, f"debug_kdj_{symbol}.csv")
    df[columns].to_csv(output_file)
    print(f"Detailed debug log saved to: {output_file}")
    print("You can open this CSV in Excel to manually verify the formulas.")
    
    # Print last 5 signals
    print("\nRecent Buy Signals:")
    print(df[df['is_buy']].tail(5)[['close', 'k', 'd', 'j']])
    
    print("\nRecent Sell Signals:")
    print(df[df['is_sell']].tail(5)[['close', 'k', 'd', 'j']])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        s = sys.argv[1]
    else:
        s = "3993.HK" # Default
    debug_kdj(s)
