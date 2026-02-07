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
    k = rsv.ewm(alpha=1 / M, adjust=False).mean()
    d = k.ewm(alpha=1 / M, adjust=False).mean()
    j = 3 * k - 2 * d

    return k, d, j


def strategy_kdj_plus(close, high, low, N=9, M=3, buy_j_th=30, sell_j_th=70):
    """
    Optimized KDJ Strategy (KDJ+)
    Long: K cross up D AND J < 30 AND Close > MA200 (Trend Filter)
    Exit: K cross down D AND J > 70
    """
    k, d, j = calculate_kdj(close, high, low, N=N, M=M)

    prev_k = k.shift(1)
    prev_d = d.shift(1)

    # MA200 Filter
    ma200 = close.rolling(window=135).mean()
    trend_filter = close > ma200

    # Cross Up: Prev K < Prev D, Current K > Current D
    gold_cross = (prev_k < prev_d) & (k > d)
    # Cross Down: Prev K > Prev D, Current K < Current D
    death_cross = (prev_k > prev_d) & (k < d)

    entries = gold_cross & (j < buy_j_th) & trend_filter
    exits = death_cross & (j > sell_j_th)

    return entries, exits


def strategy_vol_price_plus(
    close, open_, high, low, volume, ma_window=10, vol_windows=[5, 35, 135]
):
    """
    Optimized Vol-Price Strategy (VolPrice+)
    1. Relaxed Entry: Body/Total < 1/3 (was 1/6)
    2. Higher Target Exit: Close > MA60 (was MA10)
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

    # 2. K-shape (Optimized: < 0.33)
    body_len = (close - open_).abs()
    total_len = high - low
    total_len = total_len.replace(0, np.nan)

    # Relaxed condition !
    shape_condition = (body_len / total_len) < (1 / 3)

    filter_mask = vol_condition.astype(bool) & shape_condition

    # 3. Price vs MA
    ma_close = close.rolling(window=ma_window).mean()

    # Entry
    long_entries = filter_mask & (close < ma_close)

    # 4. Optimized Exit: Close > MA60
    ma60 = close.rolling(window=60).mean()
    exits = close > ma60

    return long_entries, exits


def print_stats(portfolio, name):
    total_returns = portfolio.total_return()
    mean_ret = total_returns.mean()

    # Trade-level metrics
    trade_win_rate = portfolio.trades.win_rate().mean()
    try:
        profit_factor = portfolio.trades.profit_factor().mean()
    except Exception:
        profit_factor = np.nan

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


def run_backtest():
    print("Loading data...")
    data_map = load_data(DATA_DIR, period_suffix="day")
    if not data_map:
        print("No data found.")
        return

    print(f"Aligning data for {len(data_map)} symbols...")

    def get_field(field):
        return pd.DataFrame({sym: df[field] for sym, df in data_map.items()})

    close = get_field("close")
    open_ = get_field("open")
    high = get_field("high")
    low = get_field("low")
    volume = get_field("volume")

    # CLEANING DATA
    close = close.ffill()
    open_ = open_.ffill()
    high = high.ffill()
    low = low.ffill()
    volume = volume.ffill()

    valid_mask = (close > 0) & (high > 0) & (low > 0) & close.notnull()

    # ==========================
    # 1. KDJ Plus (Trend Filter)
    # ==========================
    print("\n" + "=" * 40)
    print("Running Optimized Strategy 1: KDJ+ (with MA200 Filter)")
    print("=" * 40)

    entries_kdj, exits_kdj = strategy_kdj_plus(
        close, high, low, N=9, M=3, buy_j_th=30, sell_j_th=70
    )
    entries_kdj &= valid_mask
    exits_kdj &= valid_mask

    pf_kdj = vbt.Portfolio.from_signals(
        close=close,
        entries=entries_kdj,
        exits=exits_kdj,
        init_cash=1000000,
        fees=0.0003,
        slippage=0.0005,
        freq="1D",
    )
    print_stats(pf_kdj, "KDJ_Plus")

    # ==========================
    # 2. VolPrice Plus (Relaxed)
    # ==========================
    print("\n" + "=" * 40)
    print("Running Optimized Strategy 2: VolPrice+ (Relaxed Entry, MA60 Exit)")
    print("=" * 40)

    entries_vp, exits_vp = strategy_vol_price_plus(close, open_, high, low, volume)
    entries_vp &= valid_mask
    exits_vp &= valid_mask

    pf_vp = vbt.Portfolio.from_signals(
        close=close,
        entries=entries_vp,
        exits=exits_vp,
        init_cash=1000000,
        fees=0.0003,
        slippage=0.0005,
        freq="1D",
    )
    print_stats(pf_vp, "VolPrice_Plus")

    # ==========================
    # 3. Resonance (AND)
    # ==========================
    print("\n" + "=" * 40)
    print("Running Optimized Strategy 3: Resonance (KDJ+ AND VolPrice+)")
    print("=" * 40)

    # AND Logic for Entry
    # Note: VolPrice+ entry is rare? KDJ+ entry is filtered?
    # Let's see intersections.
    entries_res = entries_kdj & entries_vp

    # Exit: Use KDJ exit for now (or maybe either exit?)
    # Plan said "KDJ Exit"
    exits_res = exits_kdj

    pf_res = vbt.Portfolio.from_signals(
        close=close,
        entries=entries_res,
        exits=exits_res,
        init_cash=1000000,
        fees=0.0003,
        slippage=0.0005,
        freq="1D",
    )
    print_stats(pf_res, "Resonance_AND")

    # ==========================
    # 4. Trend Composite (OR + Common Filter)
    # ==========================
    print("\n" + "=" * 40)
    print(
        "Running Optimized Strategy 4: Trend Composite (MA200 Filter + OR Entry + MA60 Exit)"
    )
    print("=" * 40)

    # 1. Common Filter
    ma200 = close.rolling(window=200).mean()
    trend_filter = close > ma200

    # 2. Entries (OR)
    # Reuse raw signals from previous functions?
    # KDJ+ already has MA200, so we can use its entries.
    # VolPrice+ does not have MA200, so need to apply filter.
    entries_composite = entries_kdj | (entries_vp & trend_filter)

    # 3. Exits (MA60 trend following)
    # Close < MA60
    ma60 = close.rolling(window=60).mean()
    exits_composite = close < ma60

    entries_composite &= valid_mask
    exits_composite &= valid_mask

    pf_comp = vbt.Portfolio.from_signals(
        close=close,
        entries=entries_composite,
        exits=exits_composite,
        init_cash=1000000,
        fees=0.0003,
        slippage=0.0005,
        freq="1D",
    )
    print_stats(pf_comp, "Trend_Composite")


if __name__ == "__main__":
    run_backtest()
