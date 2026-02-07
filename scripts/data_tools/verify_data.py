
import os
import csv
import random
import time
from datetime import datetime
from decimal import Decimal
from longport.openapi import QuoteContext, Config, Period, AdjustType, HttpClient

def get_credentials():
    import json
    config_path = os.path.expanduser("~/.gemini/antigravity/mcp_config.json")
    
    app_key = os.environ.get("LONGPORT_APP_KEY")
    app_secret = os.environ.get("LONGPORT_APP_SECRET") 
    access_token = os.environ.get("LONGPORT_ACCESS_TOKEN")

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
                
                # Check for 'mcpServers' structure
                if "mcpServers" in data:
                    servers = data["mcpServers"]
                    if "longport-mcp" in servers:
                        lp_config = servers["longport-mcp"]
                        # Check 'env' sub-dictionary
                        env_config = lp_config.get("env", {})
                        if env_config:
                             # Map env vars to our variables
                             app_key = env_config.get("LONGPORT_APP_KEY", app_key)
                             app_secret = env_config.get("LONGPORT_APP_SECRET", app_secret)
                             access_token = env_config.get("LONGPORT_ACCESS_TOKEN", access_token)
                             print(f"DEBUG: Loaded credentials from mcpServers.longport-mcp.env")
                        else:
                             # Check if flat under longport-mcp (less likely for mcpServers but possible)
                             app_key = lp_config.get("app_key", app_key)
                             app_secret = lp_config.get("app_secret", app_secret)
                             access_token = lp_config.get("access_token", access_token)
                
                # Fallback: Check for flat 'longport-mcp' section (legacy/custom)
                elif "longport-mcp" in data:
                    longport_config = data["longport-mcp"]
                    if "app_key" in longport_config:
                        app_key = longport_config.get("app_key")
                        app_secret = longport_config.get("app_secret")
                        access_token = longport_config.get("access_token")

                # Update if found
                # (Logic handled above)
        except Exception as e:
            print(f"Error reading config: {e}")

    # Set env vars for SDK
    print(f"DEBUG: Using Token: {access_token[:5] if access_token else 'None'}...")
    if app_key: os.environ["LONGPORT_APP_KEY"] = app_key
    if app_secret: os.environ["LONGPORT_APP_SECRET"] = app_secret
    if access_token: os.environ["LONGPORT_ACCESS_TOKEN"] = access_token
    
    return app_key, app_secret, access_token

def verify_data():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock_data")
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist.")
        return

    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not files:
        print("No CSV files found.")
        return

    # Sample 5 random files
    samples = random.sample(files, min(5, len(files)))
    
    # Init SDK
    # HttpClient.from_env() needs env vars set
    app_key, app_secret, access_token = get_credentials()
    if not (app_key and app_secret and access_token):
         # Try to load from config file manualy if env not set
         # But usually HttpClient.from_env() is sufficient if we assume environment is ready
         pass

    # We use QuoteContext
    # Need to construct Config
    # If standard env vars are set:
    try:
        http = HttpClient.from_env() # Just to test env
        cfg = Config(os.environ["LONGPORT_APP_KEY"], os.environ["LONGPORT_APP_SECRET"], os.environ["LONGPORT_ACCESS_TOKEN"])
        ctx = QuoteContext(cfg)
    except Exception as e:
        print(f"Failed to init SDK: {e}")
        return

    print(f"Verifying {len(samples)} random files...")
    
    for filename in samples:
        symbol = filename.split("_")[0]
        filepath = os.path.join(data_dir, filename)
        
        last_row = None
        try:
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)
                if rows:
                    last_row = rows[-1]
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
            
        if not last_row:
            continue
            
        # CSV Format: Date, Open, High, Low, Close, Volume, Turnver
        csv_date = last_row[0]
        csv_close = float(last_row[4])
        
        print(f"Checking {symbol} on {csv_date} (CSV Close: {csv_close})")
        
        # Fetch from SDK
        try:
            # Fetch last 5 candles to cover potential holidays/delays
            candles = ctx.candlesticks(symbol, Period.Day, 10, AdjustType.ForwardAdjust)
            
            found = False
            for c in candles:
                # Timestamp to YYYY-MM-DD
                sdk_date = c.timestamp.strftime("%Y-%m-%d")
                
                # Careful with timezone. Longport timestamp is usually unix timestamp.
                # Let's just match date string.
                
                if sdk_date == csv_date:
                    sdk_close = float(c.close)
                    diff = abs(sdk_close - csv_close)
                    
                    if diff < 0.05: # Allow small float diff
                        print(f"  [PASS] Matches SDK Close: {sdk_close}")
                    else:
                        print(f"  [FAIL] Mismatch! SDK: {sdk_close}, Diff: {diff}")
                    found = True
                    break
            
            if not found:
                print(f"  [WARN] Date {csv_date} not found in recent SDK data (might be older than 10 days?)")
                
        except Exception as e:
            print(f"  [ERROR] SDK Request failed: {e}")

if __name__ == "__main__":
    verify_data()
