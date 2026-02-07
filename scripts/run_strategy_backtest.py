import vectorbt as vbt
import pandas as pd
import numpy as np
import os
import sys

# Add parent path to import data_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from backtesting.data_loader import load_data
except ImportError:
    # Fallback if running from scripts directory directly without package context
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from backtesting.data_loader import load_data

# Report Directory
REPORT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"
)
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock_data"
)
if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)


def calculate_kdj(close, high, low, N=9, M=3):
    """
    KDJ Calculation.
    Default: N=9, M=3.
    J = 3K - 2D
    """
    # 1. RSV
    lowest_low = low.rolling(window=N).min()
    highest_high = high.rolling(window=N).max()
    # Avoid division by zero
    denominator = highest_high - lowest_low
    denominator = denominator.replace(0, np.nan)

    rsv = (close - lowest_low) / denominator * 100

    # 2. SMA for K and D (using ewm as proxy for smooth update)
    # Pandas ewm(alpha=1/M) is similar to MMA

    k = rsv.ewm(alpha=1 / M, adjust=False).mean()
    d = k.ewm(alpha=1 / M, adjust=False).mean()
    j = 3 * k - 2 * d

    return k, d, j


def strategy_kdj(close, high, low, N=9, M=3, buy_j_th=30, sell_j_th=70):
    """
    Strategy 1: Daily KDJ
    Long: K cross up D AND J < 30
    Short: K cross down D AND J > 70
    """
    k, d, j = calculate_kdj(close, high, low, N=N, M=M)

    prev_k = k.shift(1)
    prev_d = d.shift(1)

    # Cross Up: Prev K < Prev D, Current K > Current D
    gold_cross = (prev_k < prev_d) & (k > d)
    # Cross Down: Prev K > Prev D, Current K < Current D
    death_cross = (prev_k > prev_d) & (k < d)

    entries = gold_cross & (j < buy_j_th)
    exits = death_cross & (j > sell_j_th)

    return entries, exits


def strategy_vol_price(
    close, open_, high, low, volume, ma_window=10, vol_windows=[5, 35, 135]
):
    """
    Strategy 2: Volume-Price Mean Reversion
    Filter:
      1. Volume > (at least 2 of [MA5_Vol, MA35_Vol, MA135_Vol])
      2. Abs(Close - Open) / (High - Low) < 1/6

    Long: Filter AND Close < MA10
    Short: Filter AND Close > MA10

    Exit: Opposite signal (Reversal)
    """
    # 1. Volume MA
    vol_mas = []
    for w in vol_windows:
        vol_mas.append(volume.rolling(window=w).mean())

    # Check if Volume is greater than at least 2 of the MAs
    c1 = (volume > vol_mas[0]).astype(int)
    c2 = (volume > vol_mas[1]).astype(int)
    c3 = (volume > vol_mas[2]).astype(int)

    vol_condition = (c1 + c2 + c3) >= 2

    # 2. K-shape
    body_len = (close - open_).abs()
    total_len = high - low
    total_len = total_len.replace(0, np.nan)  # avoid zero div

    shape_condition = (body_len / total_len) < (1 / 6)

    filter_mask = vol_condition.astype(bool) & shape_condition

    # 3. Price vs MA
    ma_close = close.rolling(window=ma_window).mean()

    long_entries = filter_mask & (close < ma_close)
    short_entries = filter_mask & (close > ma_close)

    return long_entries, short_entries


def run_backtest():
    print("Loading data...")
    data_map = load_data(DATA_DIR, period_suffix="day")
    if not data_map:
        print("No data found.")
        return

    print(f"Aligning data for {len(data_map)} symbols...")

    # Helper to concat
    def get_field(field):
        return pd.DataFrame({sym: df[field] for sym, df in data_map.items()})

    close = get_field("close")
    open_ = get_field("open")
    high = get_field("high")
    low = get_field("low")
    volume = get_field("volume")

    # CLEANING DATA
    # Forward fill to handle missing values within series
    close = close.ffill()
    open_ = open_.ffill()
    high = high.ffill()
    low = low.ffill()
    volume = volume.ffill()

    # Create Valid Mask (Price > 0 and not NaN)
    # VectorBT needs finite positive price for execution.
    valid_mask = (close > 0) & (high > 0) & (low > 0) & close.notnull()

    # ==========================
    # Strategy 1: KDJ
    # ==========================
    print("\n" + "=" * 40)
    print("Running Strategy 1: KDJ")
    print("=" * 40)

    entries_kdj, exits_kdj = strategy_kdj(
        close, high, low, N=9, M=3, buy_j_th=30, sell_j_th=70
    )

    # Mask Signals
    entries_kdj = entries_kdj & valid_mask
    exits_kdj = exits_kdj & valid_mask

    pf_kdj = vbt.Portfolio.from_signals(
        close=close,
        entries=entries_kdj,
        exits=exits_kdj,
        init_cash=1000000,
        fees=0.0003,
        slippage=0.0005,
        freq="1D",
    )

    print_stats(pf_kdj, "KDJ_Strategy")

    # ==========================
    # Strategy 2: Vol-Price
    # ==========================
    print("\n" + "=" * 40)
    print("Running Strategy 2: Volume-Price Mean Reversion")
    print("=" * 40)

    entries_vp, short_entries_vp = strategy_vol_price(close, open_, high, low, volume)

    # Mask Signals
    entries_vp = entries_vp & valid_mask
    short_entries_vp = short_entries_vp & valid_mask

    pf_vp = vbt.Portfolio.from_signals(
        close=close,
        entries=entries_vp,
        exits=short_entries_vp,  # Reverse logic
        init_cash=1000000,
        fees=0.0003,
        slippage=0.0005,
        freq="1D",
    )

    print_stats(pf_vp, "VolPrice_Strategy")

    # ==========================
    # Strategy 3: Or
    # ==========================
    print("\n" + "=" * 40)
    print("Running Strategy 3: Volume-Price Mean Reversion Or KDJ")
    print("=" * 40)

    # Mask Signals
    entries_vp = entries_vp | entries_kdj
    short_entries_vp = short_entries_vp | exits_kdj

    pf_vp = vbt.Portfolio.from_signals(
        close=close,
        entries=entries_vp,
        exits=short_entries_vp,  # Reverse logic
        init_cash=1000000,
        fees=0.0003,
        slippage=0.0005,
        freq="1D",
    )

    print_stats(pf_vp, "Or_Strategy")


def print_stats(portfolio, name):
    total_returns = portfolio.total_return()
    mean_ret = total_returns.mean()

    # Trade-level metrics (averaged across all symbols)
    # VBT 0.24+: portfolio.trades.win_rate(), portfolio.trades.profit_factor()
    # Check if 'trades' accessor is needed.
    # Usually portfolio.stats() calculates these.
    # portfolio.total_trades() is shortcut for portfolio.trades.count()

    trade_win_rate = portfolio.trades.win_rate().mean()
    profit_factor = portfolio.trades.profit_factor().mean()
    total_trades = portfolio.trades.count().mean()
    max_dd = portfolio.max_drawdown().mean()

    print(f"[{name}]")
    print(f"Mean Total Return:   {mean_ret*100:.2f}%")
    print(f"Avg Trade Win Rate:  {trade_win_rate*100:.2f}%")
    print(f"Avg Profit Factor:   {profit_factor:.2f}")
    print(f"Avg Total Trades:    {total_trades:.2f}")
    print(f"Avg Max Drawdown:    {max_dd*100:.2f}%")

    print(f"Max Return:          {total_returns.max()*100:.2f}%")
    print(f"Min Return:          {total_returns.min()*100:.2f}%")

    # Save Histogram
    hist_fig = total_returns.vbt.histplot(title=f"{name} Return Distribution")
    hist_fig.write_html(os.path.join(REPORT_DIR, f"{name}_hist.html"))
    print(f"Saved {name}_hist.html")

    # Save Top 5 Stats
    top_5 = total_returns.sort_values(ascending=False).head(5)
    print(f"Top 5 {name}:")
    print(top_5 * 100)

    stats = portfolio[top_5.index].stats()
    stats.to_csv(os.path.join(REPORT_DIR, f"{name}_top5_stats.csv"))
    print(f"Saved {name}_top5_stats.csv")


if __name__ == "__main__":
    run_backtest()
