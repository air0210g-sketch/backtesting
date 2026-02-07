
import json
import os
import csv
import argparse
import sys
from datetime import datetime
try:
    from longport.openapi import QuoteContext, Config, Period, AdjustType
except ImportError:
    print("Error: 'longport' library not found. Please run: pip install longport")
    sys.exit(1)

# Default path to MCP config, can be overridden env var or argument if needed in future
# For now, we assume standard location or relative to user home
CONFIG_PATH = os.path.expanduser("~/.gemini/antigravity/mcp_config.json")

def get_credentials(config_path):
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
        
    with open(config_path, 'r') as f:
        try:
            data = json.load(f)
            # Traverse to find the env vars. Assuming standard structure.
            env = data.get("mcpServers", {}).get("longport-mcp", {}).get("env", {})
            
            app_key = env.get("LONGPORT_APP_KEY")
            app_secret = env.get("LONGPORT_APP_SECRET")
            access_token = env.get("LONGPORT_ACCESS_TOKEN")
            
            if not all([app_key, app_secret, access_token]):
                print("Error: Missing LONGPORT credentials in mcp_config.json")
                sys.exit(1)
                
            return app_key, app_secret, access_token
        except json.JSONDecodeError:
            print("Error: Failed to parse mcp_config.json")
            sys.exit(1)

def map_period(period_str):
    mapping = {
        "1m": Period.Min_1,
        "5m": Period.Min_5,
        "15m": Period.Min_15,
        "30m": Period.Min_30,
        "60m": Period.Min_60,
        "day": Period.Day,
        "week": Period.Week,
        "month": Period.Month,
        "year": Period.Year,
    }
    return mapping.get(period_str.lower(), Period.Day)


def process_symbol(ctx, symbol, count, period_enum, output_dir, period_str):
    print(f"Fetching {count} {period_str} candles for {symbol}...")
    try:
        candlesticks = ctx.candlesticks(symbol, period_enum, count, AdjustType.ForwardAdjust)
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{symbol}_{period_str}.csv")
    
    # 1. Load existing data
    data_map = {} # Key: timestamp_str, Value: row_list
    headers = ["date", "open", "high", "low", "close", "volume", "turnover"]
    
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", newline="") as f:
                reader = csv.reader(f)
                file_headers = next(reader, None)
                if file_headers == headers:
                    for row in reader:
                        if row:
                            data_map[row[0]] = row
        except Exception as e:
            print(f"Warning: Failed to read existing file: {e}. Will overwrite.")
    
    # 2. Merge new data
    new_count = 0
    updated_count = 0
    
    for c in candlesticks:
        # Format timestamp consistent with existing data
        if period_str in ["day", "week", "month", "year"]:
             date_str = c.timestamp.strftime("%Y-%m-%d")
        else:
             date_str = c.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        row_data = [
            date_str,
            str(c.open),
            str(c.high),
            str(c.low),
            str(c.close),
            str(c.volume),
            str(c.turnover)
        ]
        
        if date_str in data_map:
            # Check if data actually changed
            if data_map[date_str] != row_data:
                updated_count += 1
                data_map[date_str] = row_data
        else:
            new_count += 1
            data_map[date_str] = row_data

    # 3. Write back sorted
    sorted_keys = sorted(data_map.keys())
    
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for date_key in sorted_keys:
            writer.writerow(data_map[date_key])
            
    print(f"Processed {symbol}: Total {len(sorted_keys)}. New: {new_count}. Updated: {updated_count}.")

def get_watchlist_symbols(http_client, group_names=None):
    try:
        resp = http_client.request("GET", "/v1/watchlist/groups")
        if not resp or "groups" not in resp:
            pass
        
        groups = resp if isinstance(resp, list) else resp.get('groups', [])
        
        symbols = []
        target_groups = []
        if group_names:
            # Normalize to list of lower-case strings for comparison
            target_groups = [g.strip().lower() for g in group_names]
            
        for g in groups:
            g_name = g.get('name', '')
            # If targets specified, filter.
            if target_groups and "all" not in target_groups:
                 if g_name.lower() not in target_groups:
                     continue
            
            print(f"  Found group: {g_name}, extracting symbols...")     
            for sec in g.get('securities', []):
                sym = sec.get('symbol')
                if sym:
                    symbols.append(sym)
                    
        return list(set(symbols)) # Dedup
    except Exception as e:
        print(f"Error fetching watchlist: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Download historical candlesticks from Longport.")
    parser.add_argument("--symbol", help="Stock symbol (e.g., 700.HK, AAPL.US)")
    parser.add_argument("--period", default="day", help="Time period (1m, 5m, 30m, day, week, month). Default: day")
    parser.add_argument("--count", type=int, default=1000, help="Number of candles to fetch. Default: 1000")
    parser.add_argument("--output-dir", default="stock_data", help="Directory to save CSV. Default: stock_data")
    parser.add_argument("--watchlist", help="Watchlist group names (comma separated) to download. Default: '观察,esg 50'")
    
    args = parser.parse_args()
    
    app_key, app_secret, access_token = get_credentials(CONFIG_PATH)
    
    # Setup env for HttpClient
    os.environ["LONGPORT_APP_KEY"] = app_key
    os.environ["LONGPORT_APP_SECRET"] = app_secret
    os.environ["LONGPORT_ACCESS_TOKEN"] = access_token
    
    cfg = Config(app_key, app_secret, access_token)
    ctx = QuoteContext(cfg)
    period_enum = map_period(args.period)
    
    targets = []
    
    # Logic:
    # 1. If symbol provided -> use symbol
    # 2. If watchlist provided -> use watchlist
    # 3. If neither -> use Default Watchlists ["观察", "esg 50"]
    
    if args.symbol:
        targets = [args.symbol]
    else:
        from longport.openapi import HttpClient
        http_client = HttpClient.from_env()
        
        group_list = ["观察", "esg 50"] # Default
        if args.watchlist:
            group_list = args.watchlist.split(',')
            
        print(f"Fetching watchlist groups: {group_list}...")
        targets = get_watchlist_symbols(http_client, group_list)
        print(f"Found {len(targets)} symbols in selected groups.")
        
    for sym in targets:
        process_symbol(ctx, sym, args.count, period_enum, args.output_dir, args.period)


if __name__ == "__main__":
    main()
