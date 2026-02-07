import os
import sys
import pandas as pd
from decimal import Decimal

# Ensure we can import from local backtesting package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Report Directory
REPORT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"
)
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock_data"
)
if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

from backtesting.engine import Backtester, SimpleExecutionModel
from backtesting.data_loader import load_data
from backtesting.strategies.kdj_strategy import KDJStrategy


def run_backtest():
    # 1. Load Data
    print("Loading data...")
    # Load all 'day' data from stock_data
    # Filter for a few symbols to keep demo fast or readable?
    # Let's verify '700.HK' (Tencent) specifically if available, or just load all.
    data_map = load_data(DATA_DIR, period_suffix="day")

    if not data_map:
        print(
            "No data found in stock_data/ folder. Please run download_candles.py first."
        )
        return

    print(f"Loaded {len(data_map)} symbols.")

    # 2. Setup Framework
    print("Initializing Backtester...")
    # Init Capital 1M
    initial_capital = Decimal("1000000")

    strategy = KDJStrategy(n=9, k_period=3, d_period=3, buy_qty=500)

    # Commission: 0.03% (Standard HK/CN roughly)
    execution_model = SimpleExecutionModel(slippage_bps=5, commission_rate=0.0003)

    backtester = Backtester(
        strategy=strategy,
        execution_model=execution_model,
        initial_capital=initial_capital,
    )

    # 3. Run
    print("Running Backtest...")
    try:
        results = backtester.run(data_map)
    except Exception as e:
        print(f"Runtime Error: {e}")
        import traceback

        traceback.print_exc()
        return

    # 4. Analyze Results
    print("\n" + "=" * 40)
    print("BACKTEST RESULTS")
    print("=" * 40)

    end_equity = backtester.portfolio.get_equity(
        {
            sym: Decimal(str(df.iloc[-1]["close"]))
            for sym, df in data_map.items()
            if not df.empty
        }
    )
    total_return = (end_equity - initial_capital) / initial_capital * 100

    print(f"Initial Capital: {initial_capital:,.2f}")
    print(f"Final Equity:    {end_equity:,.2f}")
    print(f"Total Return:    {total_return:.2f}%")

    trades = backtester.trades
    print(f"Total Trades:    {len(trades)}")

    if trades:
        # Simple Trade Analysis
        # Note: 'trades' here are Fills (individual executions), not Round-Trip trades.
        # Approximating win rate from realized PnL of positions is better,
        # but Portfolio tracks realized PnL per symbol.

        total_realized_pnl = Decimal("0")
        for sym, pos in backtester.portfolio.positions.items():
            total_realized_pnl += pos.realized_pnl

        print(f"Realized PnL:    {total_realized_pnl:,.2f}")

    print("\nPositions:")
    for sym, pos in backtester.portfolio.positions.items():
        if pos.quantity != 0:
            print(f"  {sym}: {pos.quantity} shares")

    # Save Equity Curve
    if not results.empty:
        results.to_csv(os.path.join(REPORT_DIR, "backtest_results.csv"))
        print("\nEquity curve saved to 'backtest_results.csv'")
    else:
        print("\nNo results generated (Did data align?)")


if __name__ == "__main__":
    run_backtest()
