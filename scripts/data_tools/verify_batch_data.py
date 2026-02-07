import os
import json
import pandas as pd
import random
import sys

# Configuration
STOCK_DATA_DIR = "stock_data"
STOCK_LIST_PATH = os.path.join(STOCK_DATA_DIR, "stock_list.json")


def verify_data():
    if not os.path.exists(STOCK_LIST_PATH):
        print(f"Error: {STOCK_LIST_PATH} not found.")
        return

    with open(STOCK_LIST_PATH, "r") as f:
        symbols = json.load(f)

    if not symbols:
        print("Error: stock_list.json is empty.")
        return

    # Randomly sample 10 symbols
    sample_size = min(10, len(symbols))
    sampled_symbols = random.sample(symbols, sample_size)
    print(f"Sampled 10 symbols for verification: {sampled_symbols}")
    print("-" * 50)

    results = []
    for sym in sampled_symbols:
        filename = f"{sym}_day.csv"
        filepath = os.path.join(STOCK_DATA_DIR, filename)

        status = {
            "Symbol": sym,
            "File Exists": False,
            "Rows": 0,
            "Nulls": 0,
            "Last Date": "N/A",
            "Integrity": "FAIL",
        }

        if os.path.exists(filepath):
            status["File Exists"] = True
            try:
                df = pd.read_csv(filepath)
                status["Rows"] = len(df)
                status["Nulls"] = df.isnull().sum().sum()
                if not df.empty:
                    status["Last Date"] = df.iloc[-1]["date"]
                    # Basic integrity: Has headers, has rows, no nulls
                    if len(df) > 10 and status["Nulls"] == 0:
                        status["Integrity"] = "PASS"
            except Exception as e:
                status["Integrity"] = f"ERROR: {str(e)}"

        results.append(status)

    # Output as Table
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    # Summary
    success_count = sum(1 for r in results if r["Integrity"] == "PASS")
    print("-" * 50)
    print(f"Verification Summary: {success_count}/{sample_size} Passed.")


if __name__ == "__main__":
    verify_data()
