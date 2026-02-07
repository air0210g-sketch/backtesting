import pandas as pd
import numpy as np


def calc_kdj(close, high, low, N=9, M=3):
    """
    KDJ Calculation.
    Default: N=9, M=3.
    J = 3K - 2D
    Returns: k, d, j (all pd.Series or DataFrame)
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


def get_kdj_cross_signals(k, d, j, threshold=None, mode="long"):
    """
    Generate KDJ Cross Signals.

    Args:
        threshold (float): Filtering threshold for J.
                           If mode='long', requires J < threshold (e.g. 35).
                           If mode='short', requires J > threshold (e.g. 80).
        mode (str): 'long' (Gold Cross) or 'short' (Death Cross).

    Returns:
        pd.DataFrame/Series: Boolean signal mask.
    """
    prev_k = k.shift(1)
    prev_d = d.shift(1)

    if mode == "long":
        # Gold Cross: Prev K < Prev D  AND  K > D
        cross = (prev_k < prev_d) & (k >= d)
        if threshold is not None:
            # Filter: J < Threshold (OverSold)
            condition = cross & (j <= threshold)
        else:
            condition = cross

    elif mode == "short":
        # Death Cross: Prev K > Prev D  AND  K < D
        cross = (prev_k > prev_d) & (k <= d)
        if threshold is not None:
            # Filter: J > Threshold (OverBought)
            condition = cross & (j >= threshold)
        else:
            condition = cross
    else:
        raise ValueError("mode must be 'long' or 'short'")

    return condition


def resample_to_weekly(open_df, high_df, low_df, close_df):
    """
    Resample daily dataframes to weekly.
    Assumes index is datetime.
    """
    # rule='W-FRI' maps to Week ending on Friday
    w_open = open_df.resample("W-FRI").first()
    w_high = high_df.resample("W-FRI").max()
    w_low = low_df.resample("W-FRI").min()
    w_close = close_df.resample("W-FRI").last()

    return w_open, w_high, w_low, w_close


def calc_weekly_kdj(open_daily, high_daily, low_daily, close_daily, N=9, M=3):
    """
    Calculate Weekly KDJ and align it back to Daily index (ffill).
    """
    # 1. Resample
    w_open, w_high, w_low, w_close = resample_to_weekly(
        open_daily, high_daily, low_daily, close_daily
    )

    # 2. Calculate Weekly KDJ
    wk, wd, wj = calc_kdj(w_close, w_high, w_low, N, M)

    # 3. Realign to Daily (Forward Fill)
    # Use reindex to map weekly values to daily dates
    # method='ffill' propagates the last known weekly value forward
    dk = wk.reindex(close_daily.index, method="ffill")
    dd = wd.reindex(close_daily.index, method="ffill")
    dj = wj.reindex(close_daily.index, method="ffill")

    return dk, dd, dj


def check_volume_agv_breakout(volume, multiple=1.0):
    """
    Check if current volume is > multiple * MovingAverage(volume, window).
    Returns True if volume breaks out of at least 2 out of 3 moving averages (5, 35, 135).
    """
    vol_ma5 = volume.rolling(window=5).mean().ffill()
    vol_ma35 = volume.rolling(window=35).mean().ffill()
    vol_ma135 = volume.rolling(window=135).mean().ffill()

    # Vectorized counting of conditions
    count = (
        (volume > (vol_ma5 * multiple)).astype(int)
        + (volume > (vol_ma35 * multiple)).astype(int)
        + (volume > (vol_ma135 * multiple)).astype(int)
    )

    return count >= 2


def check_volume_breakout(volume, window=20, multiple=2.0):
    """
    Check if current volume is > multiple * MovingAverage(volume, window).
    """
    vol_ma = volume.rolling(window=window).mean()
    is_breakout = volume > (vol_ma * multiple)
    return is_breakout


try:
    import talib
except ImportError:
    talib = None


def get_candle_patterns(open, high, low, close, patterns=["CDLENGULFING", "CDLHAMMER"]):
    """
    Detect multiple candlestick patterns.

    Args:
        patterns (list): List of TA-Lib pattern function names (e.g. 'CDLENGULFING').

    Returns:
        pd.DataFrame: DataFrame where columns are pattern names and values are signals.
                      (100=Bullish, -100=Bearish, 0=None).
        pd.Series: 'any_bullish' mask (True if any pattern is 100).
        pd.Series: 'any_bearish' mask (True if any pattern is -100).
    """
    if talib is None:
        print("Warning: TA-Lib not installed. Cannot detect patterns.")
        return (
            pd.DataFrame(),
            pd.Series(False, index=close.index),
            pd.Series(False, index=close.index),
        )

    # Initialize mask dataframes
    bullish_mask = pd.DataFrame(False, index=close.index, columns=close.columns)
    bearish_mask = pd.DataFrame(False, index=close.index, columns=close.columns)

    # Iterate over symbols (columns)
    for col in close.columns:
        try:
            # Extract series as numpy arrays (handle float vs obj)
            o_s = open[col].values.astype(float)
            h_s = high[col].values.astype(float)
            l_s = low[col].values.astype(float)
            c_s = close[col].values.astype(float)

            # Temporary storage for this symbol
            symbol_bullish = np.zeros(len(close), dtype=bool)
            symbol_bearish = np.zeros(len(close), dtype=bool)

            for pat in patterns:
                if hasattr(talib, pat):
                    func = getattr(talib, pat)
                    res = func(o_s, h_s, l_s, c_s)

                    symbol_bullish |= res == 100
                    symbol_bearish |= res == -100

            bullish_mask[col] = symbol_bullish
            bearish_mask[col] = symbol_bearish

        except Exception as e:
            # print(f"Error on {col}: {e}")
            pass

    # Return structure:
    # signals (None, deprecated for multi-symbol),
    # any_bullish (DataFrame mask),
    # any_bearish (DataFrame mask)
    return None, bullish_mask, bearish_mask
