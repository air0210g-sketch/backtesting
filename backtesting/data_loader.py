import os
import pandas as pd
from typing import Dict, List, Optional

def load_data(data_dir: str, symbols: Optional[List[str]] = None, period_suffix: str = "day") -> Dict[str, pd.DataFrame]:
    """
    Load CSV data from data_dir.
    files are expected to be named: {symbol}_{period}.csv
    """
    data_store = {}
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found.")
        return {}

    files = os.listdir(data_dir)
    
    for f in files:
        if not f.endswith(".csv"):
            continue
            
        # Filename format: SYMBOL_PERIOD.csv (e.g. 700.HK_day.csv)
        # We need to match period_suffix if provided
        if period_suffix and not f.endswith(f"_{period_suffix}.csv"):
             continue

        # Extract symbol
        # Assumption: symbol is everything before the last underscore
        parts = f.rsplit("_", 1)
        if len(parts) != 2:
            continue
            
        symbol = parts[0]
        
        if symbols and symbol not in symbols:
            continue
            
        file_path = os.path.join(data_dir, f)
        
        try:
            df = pd.read_csv(file_path)
            # Ensure columns are lower case
            df.columns = [c.lower() for c in df.columns]
            
            # Parse date
            # Assuming 'date' column exists
            if 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # Normalize types
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                data_store[symbol] = df
            else:
                print(f"Skipping {f}: No 'date' column found.")
                
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    return data_store

import numpy as np

def align_and_clean_data(data_map: Dict[str, pd.DataFrame]):
    """
    Aligns data from multiple symbols into wide DataFrames (Time x Symbol) and enforces data quality.
    
    Cleaning steps:
    1. Sort index.
    2. Replace 0 values with NaN (prices cannot be 0).
    3. Forward fill (propagate last valid price).
    4. Back fill (fill initial gaps with first valid price).
    5. Fallback fill with 1.0 to ensure no NaNs remain (which crash vectorbt).
    6. Ensure strictly positive values.
    
    Returns:
        tuple: (open_df, high_df, low_df, close_df, volume_df)
    """
    if not data_map:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print(f"正在对齐 {len(data_map)} 只股票的数据...")
    
    # Create aligned DataFrames
    close = pd.DataFrame({sym: df['close'] for sym, df in data_map.items()})
    open_p = pd.DataFrame({sym: df['open'] for sym, df in data_map.items()}) # 'open' kwarg conflict
    high = pd.DataFrame({sym: df['high'] for sym, df in data_map.items()})
    low = pd.DataFrame({sym: df['low'] for sym, df in data_map.items()})
    volume = pd.DataFrame({sym: df['volume'] for sym, df in data_map.items()})
    
    # Sort Index
    for df in [open_p, high, low, close, volume]:
        df.sort_index(inplace=True)

    def clean_price_df(df, name="price"):
        # 1. Replace 0 with NaN
        if (df == 0).any().any():
            # print(f"  [Cleaning] Found 0 in {name}, setting to NaN.")
            df = df.replace(0, np.nan)
        
        # 2. Fill gaps
        # bfill first to handle listing delays (use listing price for pre-listing logic if needed, 
        # or just have valid data). 
        # CAUTION: bfill creates look-ahead bias if we trade before listing. 
        # But for 'finite' checks, we need values.
        # Ideally we mask signals, but here we just ensure price validity.
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 3. Fallback for remaining NaNs (if any columns are empty or full of 0s)
        if df.isna().any().any():
            # print(f"  [Cleaning] Found remaining NaNs in {name}, filling with 1.0.")
            df = df.fillna(1.0)
            
        # 4. Final safety clamp
        if (df <= 0).any().any():
            # print(f"  [Cleaning] Found <=0 in {name} after fill, forcing to 1.0.")
            df[df <= 0] = 1.0
            
        return df

    print("正在清洗价格数据 (步骤: 0->NaN, ffill, bfill, 填充 1.0)...")
    open_p = clean_price_df(open_p, "Open")
    high = clean_price_df(high, "High")
    low = clean_price_df(low, "Low")
    close = clean_price_df(close, "Close")
    
    # Volume doesn't crash if 0 or NaN usually, but let's clean NaNs to 0
    volume = volume.fillna(0)
    
    return open_p, high, low, close, volume
