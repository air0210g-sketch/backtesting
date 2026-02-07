import vectorbt as vbt
import pandas as pd
import numpy as np
import os
import sys

# Add parent path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from backtesting.data_loader import load_data
    import backtesting.indicators as inds
except ImportError:
    # Fallback
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from backtesting.data_loader import load_data
    import backtesting.indicators as inds

# Report Directory
REPORT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"
)
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock_data"
)
if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)


def print_stats(portfolio, name):
    total_returns = portfolio.total_return()
    mean_ret = total_returns.mean()
    trade_win_rate = portfolio.trades.win_rate().mean()
    total_trades = portfolio.trades.count().mean()
    max_dd = portfolio.max_drawdown().mean()

    print(f"[{name}]")
    print(f"Mean Total Return:   {mean_ret*100:.2f}%")
    print(f"Avg Trade Win Rate:  {trade_win_rate*100:.2f}%")
    print(f"Avg Total Trades:    {total_trades:.2f}")
    print(f"Avg Max Drawdown:    {max_dd*100:.2f}%")
    print(f"Max Return:          {total_returns.max()*100:.2f}%")

    # Save Histogram
    hist_fig = total_returns.vbt.histplot(title=f"{name} Return Distribution")
    hist_fig.write_html(os.path.join(REPORT_DIR, f"{name}_hist.html"))
    print(f"Saved {name}_hist.html")

    # Save Top 5
    top_5 = total_returns.sort_values(ascending=False).head(5)
    print(f"Top 5 {name}:")
    print(top_5 * 100)


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

    # Cleaning
    close = close.ffill()
    open_ = open_.ffill()
    high = high.ffill()
    low = low.ffill()
    volume = volume.ffill()
    valid_mask = (close > 0) & (high > 0) & (low > 0) & close.notnull()

    print("Calculating Indicators using backtesting.indicators...")

    # 1. Daily KDJ
    k, d, j = inds.calc_kdj(close, high, low)

    # 2. Weekly KDJ (Aligned to Daily)
    wk, wd, wj = inds.calc_weekly_kdj(open_, high, low, close)

    # 3. Volume Breakout
    # Volume > 2.0 * MA20
    vol_breakout = inds.check_volume_breakout(volume, window=20, multiple=2.0)

    # ==========================================
    # Strategy A: Weekly KDJ Gold Cross Only
    # ==========================================
    print("\n[Strategy A] Weekly KDJ Gold Cross (J < 35)")
    weekly_long = inds.get_kdj_cross_signals(wk, wd, wj, threshold=35, mode="long")
    # Exit on Weekly Death Cross (J > 80)
    weekly_exit = inds.get_kdj_cross_signals(wk, wd, wj, threshold=80, mode="short")

    entries_A = weekly_long & valid_mask
    exits_A = weekly_exit & valid_mask

    pf_A = vbt.Portfolio.from_signals(
        close=close,
        entries=entries_A,
        exits=exits_A,
        init_cash=1000000,
        fees=0.0003,
        slippage=0.0005,
        freq="1D",
    )
    print_stats(pf_A, "Weekly_KDJ_Trend")

    # ==========================================
    # Strategy B: Weekly Trend + Daily Vol Breakout + Daily KDJ Buy
    # "Resonance++"
    # ==========================================
    print(
        "\n[Strategy B] Resonance++ (Weekly K/D Bullish + Vol Breakout + Daily KDJ Buy)"
    )

    # Weekly Trend Filter: Weekly K > Weekly D (Bullish Zone)
    weekly_bullish = wk > wd

    # Daily Trigger:
    # 1. Daily KDJ Gold Cross (J < 40)
    daily_long_signal = inds.get_kdj_cross_signals(k, d, j, threshold=40, mode="long")

    # 2. Volume Breakout
    # We want entry when BOTH happen? or Vol breakout happens near KDJ signal?
    # Let's say: Entry = Daily KDJ Buy AND (Vol Breakout Today OR Vol Breakout Yesterday)
    # AND Weekly Trend is Up.

    vol_breakout_recent = vol_breakout | vol_breakout.shift(1)

    entries_B = daily_long_signal & vol_breakout_recent & weekly_bullish

    # Exit: Daily KDJ Death Cross (J > 80) OR Weekly Death Cross
    daily_exit = inds.get_kdj_cross_signals(k, d, j, threshold=80, mode="short")
    entries_B &= valid_mask
    exits_B = (daily_exit | weekly_exit) & valid_mask

    pf_B = vbt.Portfolio.from_signals(
        close=close,
        entries=entries_B,
        exits=exits_B,
        init_cash=1000000,
        fees=0.0003,
        slippage=0.0005,
        freq="1D",
    )
    print_stats(pf_B, "Resonance_Plus_Weekly")

    # ==========================================
    # Strategy C: Weekly Trend + Technical Pattern (Engulfing/Hammer)
    # ==========================================
    print("\n[Strategy C] Trend + Patterns (Hammer/Engulfing)")

    # 1. Weekly Trend Filter (Bullish)
    # Already computed: weekly_bullish

    # 2. Bullish Patterns
    # Identify Engulfing or Hammer
    pat_signals, any_bullish_pat, _ = inds.get_candle_patterns(
        open_,
        high,
        low,
        close,
        patterns=["CDLENGULFING", "CDLHAMMER", "CDLPIERCING", "CDLMORNINGSTAR"],
    )

    # Entry: Weekly Trend UP AND Bullish Pattern Today
    entries_C = weekly_bullish & any_bullish_pat

    # Exit: KDJ Death Cross (Daily) - Simple exit
    exits_C = inds.get_kdj_cross_signals(k, d, j, threshold=80, mode="short")

    entries_C &= valid_mask
    exits_C &= valid_mask

    pf_C = vbt.Portfolio.from_signals(
        close=close,
        entries=entries_C,
        exits=exits_C,
        init_cash=1000000,
        fees=0.0003,
        slippage=0.0005,
        freq="1D",
    )
    print_stats(pf_C, "Trend_Pattern_Strategy")


if __name__ == "__main__":
    run_backtest()
