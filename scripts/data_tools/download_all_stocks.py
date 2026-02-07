import os
import json
import csv
import sys
import time
from datetime import date, timedelta, datetime

try:
    from longport.openapi import QuoteContext, Config, Period, AdjustType
except ImportError:
    print("Error: 'longport' library not found. Please run: pip install longport")
    sys.exit(1)

# Configuration
STOCK_DATA_DIR = "stock_data"
STOCK_LIST_PATH = os.path.join(STOCK_DATA_DIR, "stock_list.json")
CONFIG_PATH = os.path.expanduser("~/.gemini/antigravity/mcp_config.json")
START_DATE = date(2014, 1, 1)
TODAY = date.today()


def get_ctx():
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Config file not found at {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH, "r") as f:
        data = json.load(f)
        # Search for longport-mcp env
        if "mcpServers" in data and "longport-mcp" in data["mcpServers"]:
            env = data["mcpServers"]["longport-mcp"].get("env", {})
        elif "longport-mcp" in data:
            env = data["longport-mcp"].get("env", data["longport-mcp"])
        else:
            print("Error: longport-mcp config not found in mcp_config.json")
            sys.exit(1)

    app_key = env.get("LONGPORT_APP_KEY")
    app_secret = env.get("LONGPORT_APP_SECRET")
    access_token = env.get("LONGPORT_ACCESS_TOKEN")

    if not all([app_key, app_secret, access_token]):
        print("Error: Missing credentials in config.")
        sys.exit(1)

    cfg = Config(app_key, app_secret, access_token)
    return QuoteContext(cfg)


def fetch_history_for_symbol(ctx, symbol, start_date=START_DATE):
    all_candles = []
    current_start = start_date

    print(f"[{symbol}] Fetching history from {start_date}...")

    while current_start < TODAY:
        # Max 1000 candles per request. 3 years of daily data is approx 750 days.
        chunk_end = min(current_start + timedelta(days=365 * 3), TODAY)

        try:
            candles = ctx.history_candlesticks_by_date(
                symbol, Period.Day, AdjustType.ForwardAdjust, current_start, chunk_end
            )
        except Exception as e:
            print(
                f"[{symbol}] Error fetching batch {current_start} -> {chunk_end}: {e}"
            )
            # Potential rate limit or network error, sleep longer
            time.sleep(1)
            # Break to avoid infinite loop on fatal error per symbol
            break

        if candles:
            all_candles.extend(candles)

        current_start = chunk_end + timedelta(days=1)
        time.sleep(0.2)  # Throttle

    return all_candles


def get_last_date(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return None
            last_row = None
            for row in reader:
                last_row = row
            if last_row:
                return datetime.strptime(last_row[0], "%Y-%m-%d").date()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return None


def merge_and_save_candles(filepath, new_candles):
    data_map = {}
    headers = ["date", "open", "high", "low", "close", "volume", "turnover"]

    if os.path.exists(filepath):
        try:
            with open(filepath, "r", newline="") as f:
                reader = csv.reader(f)
                file_headers = next(reader, None)
                if file_headers == headers:
                    for row in reader:
                        if row:
                            data_map[row[0]] = row
        except Exception as e:
            print(f"Error reading existing {filepath}: {e}")

    count_new = 0
    for c in new_candles:
        d_str = c.timestamp.strftime("%Y-%m-%d")
        row = [
            d_str,
            str(c.open),
            str(c.high),
            str(c.low),
            str(c.close),
            str(c.volume),
            str(c.turnover),
        ]
        if d_str not in data_map:
            count_new += 1
        data_map[d_str] = row

    sorted_dates = sorted(data_map.keys())
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for d in sorted_dates:
            writer.writerow(data_map[d])
    return count_new


def main():
    if not os.path.exists(STOCK_LIST_PATH):
        print(f"Error: {STOCK_LIST_PATH} not found.")
        return

    with open(STOCK_LIST_PATH, "r") as f:
        symbols = json.load(f)

    print(f"Read {len(symbols)} symbols from {STOCK_LIST_PATH}.")
    ctx = get_ctx()

    for i, sym in enumerate(symbols):
        filepath = os.path.join(STOCK_DATA_DIR, f"{sym}_day.csv")
        last_date = get_last_date(filepath)

        start_fetch_date = START_DATE
        if last_date:
            days_diff = (TODAY - last_date).days
            if days_diff <= 1:
                print(f"[{i+1}/{len(symbols)}] {sym}: Already up to date.")
                continue
            start_fetch_date = last_date + timedelta(days=1)
            print(f"[{i+1}/{len(symbols)}] {sym}: Incremental from {start_fetch_date}")
        else:
            print(f"[{i+1}/{len(symbols)}] {sym}: Full download from {START_DATE}")

        candles = fetch_history_for_symbol(ctx, sym, start_date=start_fetch_date)
        if candles:
            added = merge_and_save_candles(filepath, candles)
            print(
                f"  Processed {sym}. Added {added} new records. Total: {len(candles) if not last_date else 'updated'}"
            )
        else:
            print(f"  No new data for {sym}.")

    print("\nBatch download complete.")


if __name__ == "__main__":
    main()
