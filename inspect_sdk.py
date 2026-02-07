

import os
import json
import sys
from datetime import date
from longport.openapi import QuoteContext, Config, Period, AdjustType

# Load Config
CONFIG_PATH = os.path.expanduser("~/.gemini/antigravity/mcp_config.json")
with open(CONFIG_PATH, 'r') as f:
    data = json.load(f)
    env = data.get("mcpServers", {}).get("longport-mcp", {}).get("env", {})
    
app_key = env.get("LONGPORT_APP_KEY")
app_secret = env.get("LONGPORT_APP_SECRET")
access_token = env.get("LONGPORT_ACCESS_TOKEN")

cfg = Config(app_key, app_secret, access_token)
ctx = QuoteContext(cfg)

print("Fetching history...")
try:
    # 10 years roughly
    candles = ctx.history_candlesticks_by_date("23240.HK", Period.Day, AdjustType.ForwardAdjust, date(2014, 1, 1), date(2024, 2, 1))
    print(f"Fetched {len(candles)} candles.")
except Exception as e:
    print(f"Error: {e}")

