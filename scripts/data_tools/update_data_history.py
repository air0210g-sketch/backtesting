import os
import json
import csv
import sys
import time
from datetime import date, timedelta, datetime
from longport.openapi import QuoteContext, Config, Period, AdjustType

# Configuration
STOCK_DATA_DIR = "stock_data"
CONFIG_PATH = os.path.expanduser("~/.gemini/antigravity/mcp_config.json")
START_DATE = date(2014, 1, 1)
TODAY = date.today()


def get_ctx():
    with open(CONFIG_PATH, "r") as f:
        data = json.load(f)
        env = data.get("mcpServers", {}).get("longport-mcp", {}).get("env", {})

    app_key = env.get("LONGPORT_APP_KEY")
    app_secret = env.get("LONGPORT_APP_SECRET")
    access_token = env.get("LONGPORT_ACCESS_TOKEN")

    cfg = Config(app_key, app_secret, access_token)
    return QuoteContext(cfg)


def fetch_history_for_symbol(ctx, symbol, start_date=START_DATE):
    all_candles = []
    current_start = start_date

    print(f"[{symbol}] Fetching history from {start_date}...")

    while current_start < TODAY:
        # Optimize: Batch 3 years (approx 750 trading days < 1000 limit)
        # 1000 limit is strict, so 3 years is safe buffer.
        chunk_end = min(current_start + timedelta(days=365 * 3), TODAY)

        try:
            # Fetch batch for this chunk
            # Note: history_candlesticks_by_date is inclusive of start/end usually.
            # print(f"  Fetching range: {current_start} -> {chunk_end}")
            candles = ctx.history_candlesticks_by_date(
                symbol, Period.Day, AdjustType.ForwardAdjust, current_start, chunk_end
            )
        except Exception as e:
            print(
                f"[{symbol}] Error fetching batch {current_start} -> {chunk_end}: {e}"
            )
            # Move on to avoid infinite loop on error
            current_start = chunk_end + timedelta(days=1)
            continue

        if candles:
            all_candles.extend(candles)
            # print(f"    Fetched {len(candles)} candles.")

        # Advance current_start for next chunk
        current_start = chunk_end + timedelta(days=1)

        # Avoid rate limits
        time.sleep(0.2)

    return all_candles


def get_last_date(filepath):
    """
    Get the last date from the CSV file.
    Returns: date object or None
    """
    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, "r") as f:
            # Read last line efficiently? Or just read all using pandas (easier but slower for huge files)
            # Given files are small (< 1MB), pandas is fine, or just csv reader.
            # Let's use csv reader to get last line.
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return None

            last_row = None
            for row in reader:
                last_row = row

            if last_row:
                # date is first column
                return datetime.strptime(last_row[0], "%Y-%m-%d").date()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    return None


def merge_and_save_candles(filepath, new_candles):
    """
    Merge new candles with existing file content, dedup, sort, and save.
    """
    # 1. Load existing
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

    # 2. Merge new
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

    # 3. Save
    sorted_dates = sorted(data_map.keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for d in sorted_dates:
            writer.writerow(data_map[d])

    return count_new


def save_candles(filepath, candles):
    """全量写入 K 线到 CSV（用于 FULL 模式）。"""
    headers = ["date", "open", "high", "low", "close", "volume", "turnover"]
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for c in candles:
            writer.writerow([
                c.timestamp.strftime("%Y-%m-%d"),
                str(c.open),
                str(c.high),
                str(c.low),
                str(c.close),
                str(c.volume),
                str(c.turnover),
            ])


def run_update(data_dir=None):
    """
    增量更新日线 CSV。可被外部直接调用。
    data_dir: 数据目录路径，默认 None 表示使用脚本内 STOCK_DATA_DIR。
    """
    if data_dir:
        effective_dir = os.path.abspath(data_dir)
    else:
        _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        effective_dir = os.path.join(_root, STOCK_DATA_DIR)
    if not os.path.isdir(effective_dir):
        print(f"Directory {effective_dir} not found.")
        return

    try:
        ctx = get_ctx()
    except Exception as e:
        print(f"无法初始化 LongPort 连接: {e}")
        raise

    files = [f for f in os.listdir(effective_dir) if f.endswith("_day.csv")]
    symbols = [f.replace("_day.csv", "") for f in files]
    print(f"Found {len(symbols)} symbols to check.")

    for i, sym in enumerate(symbols):
        filepath = os.path.join(effective_dir, f"{sym}_day.csv")
        last_date = get_last_date(filepath)

        start_fetch_date = START_DATE
        mode = "FULL"

        if last_date:
            days_diff = (TODAY - last_date).days
            if days_diff <= 180:
                mode = "INCREMENTAL"
                start_fetch_date = last_date + timedelta(days=1)
            else:
                mode = "FULL (Gap > 180 days)"
        else:
            mode = "FULL (New/Error)"

        print(f"[{i+1}/{len(symbols)}] {sym}: {mode}. Last Date: {last_date}")

        if start_fetch_date >= TODAY:
            print(f"  Already up to date.")
            continue

        candles = fetch_history_for_symbol(ctx, sym, start_date=start_fetch_date)

        if candles:
            if mode.startswith("INCREMENTAL"):
                added = merge_and_save_candles(filepath, candles)
                print(f"  Appended {added} new records.")
            else:
                save_candles(filepath, candles)
                print(f"  Overwritten with {len(candles)} records.")
        else:
            print(f"  No new data fetched.")

    print("All done.")


def main():
    if not os.path.exists(STOCK_DATA_DIR):
        print(f"Directory {STOCK_DATA_DIR} not found.")
        return

    ctx = get_ctx()

    files = [f for f in os.listdir(STOCK_DATA_DIR) if f.endswith("_day.csv")]
    symbols = [f.replace("_day.csv", "") for f in files]

    print(f"Found {len(symbols)} symbols to check.")

    for i, sym in enumerate(symbols):
        filepath = os.path.join(STOCK_DATA_DIR, f"{sym}_day.csv")
        last_date = get_last_date(filepath)

        start_fetch_date = START_DATE
        mode = "FULL"

        if last_date:
            days_diff = (TODAY - last_date).days
            if days_diff <= 180:
                # Incremental Update
                mode = "INCREMENTAL"
                start_fetch_date = last_date + timedelta(days=1)
            else:
                # Gap too large, Full Update
                mode = "FULL (Gap > 180 days)"
        else:
            mode = "FULL (New/Error)"

        print(f"[{i+1}/{len(symbols)}] {sym}: {mode}. Last Date: {last_date}")

        if start_fetch_date >= TODAY:
            print(f"  Already up to date.")
            continue

        # Fetch logic (adapted to use start_fetch_date)
        # We need to temporarily modify global START_DATE or pass it to fetch function.
        # Let's refactor fetch_history_for_symbol to accept start_date

        # Refactored fetch call:
        candles = fetch_history_for_symbol(ctx, sym, start_date=start_fetch_date)

        if candles:
            if mode.startswith("INCREMENTAL"):
                added = merge_and_save_candles(filepath, candles)
                print(f"  Appended {added} new records.")
            else:
                save_candles(sym, candles)  # Overwrite
                print(f"  Overwritten with {len(candles)} records.")
        else:
            print(f"  No new data fetched.")

    print("All done.")


# We need to update fetch_history_for_symbol signature as well in the same file
# But replace_file_content is block based.
# Let's just update the whole file logic or use multi_replace if needed?
# The fetch function is above main. I should probably rewrite fetch_history_for_symbol too.

if __name__ == "__main__":
    main()
