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


def calc_obv(close, volume):
    """
    On-Balance Volume (OBV).
    OBV[i] = OBV[i-1] + sign(close[i] - close[i-1]) * volume[i]，sign(0)=0。
    用于趋势确认与背离：价创新高但 OBV 未创新高视为量价背离（假突破/噪音）。
    """
    delta = close.diff()
    sign = np.sign(delta)
    sign = sign.fillna(0)
    obv = (sign * volume).cumsum()
    return obv


def zigzag_pivots(high, low, close, depth_pct=0.05):
    """
    基于百分比的 ZigZag，得到波峰波谷序列（过滤短线噪音）。
    depth_pct: 反转深度，如 0.05 表示 5%。
    返回: list of (index, value, is_high)，按时间序，高低交替。
    """
    if hasattr(high, "values"):
        high = high.values
    if hasattr(low, "values"):
        low = low.values
    if hasattr(close, "values"):
        close = close.values
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    n = len(high)
    if n == 0:
        return []
    pivots = []
    extreme = high[0]
    extreme_idx = 0
    direction = "down"  # 当前在找从高点到低点的回落，下一拐点为低点

    for i in range(1, n):
        if direction == "down":
            if low[i] <= extreme * (1.0 - depth_pct):
                pivots.append((extreme_idx, float(extreme), True))
                extreme = low[i]
                extreme_idx = i
                direction = "up"
            elif high[i] > extreme:
                extreme = high[i]
                extreme_idx = i
        else:
            if high[i] >= extreme * (1.0 + depth_pct):
                pivots.append((extreme_idx, float(extreme), False))
                extreme = high[i]
                extreme_idx = i
                direction = "down"
            elif low[i] < extreme:
                extreme = low[i]
                extreme_idx = i

    if extreme_idx is not None:
        pivots.append((extreme_idx, float(extreme), direction == "up"))

    return pivots


def zigzag_pivots_mt4(high, low, close, depth=12, deviation=0.05, backstep=3):
    """
    MT4 风格 ZigZag 拐点序列。
    参数（与 MT4 默认一致）:
      depth: 两个拐点 bar 之间最少间隔根数，过滤过密拐点。
      deviation: 确认反转的最小价格变化比例，如 0.05 表示 5%。
      backstep: 同向极值更新时的最小间隔（如连续更高高点，仅当新高点 bar 距上一候选 bar ≥ backstep 才替换候选），用于合并邻近极值。
    返回: list of (index, value, is_high)，按时间序，高低交替。
    """
    if hasattr(high, "values"):
        high = high.values
    if hasattr(low, "values"):
        low = low.values
    if hasattr(close, "values"):
        close = close.values
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    n = len(high)
    if n <= depth:
        return []

    pivots = []
    last_bar = -depth - 1
    last_val = low[0]
    direction = "up"
    cand_bar = 0
    cand_val = high[0]

    for i in range(1, n):
        if direction == "up":
            if high[i] > cand_val:
                if cand_bar is None or (i - cand_bar) >= backstep:
                    cand_bar = i
                    cand_val = high[i]
            if low[i] <= cand_val * (1.0 - deviation):
                if len(pivots) == 0 or (cand_bar - last_bar) >= depth:
                    pivots.append((cand_bar, float(cand_val), True))
                    last_bar = cand_bar
                    last_val = cand_val
                direction = "down"
                cand_bar = i
                cand_val = low[i]
        else:
            if low[i] < cand_val:
                if cand_bar is None or (i - cand_bar) >= backstep:
                    cand_bar = i
                    cand_val = low[i]
            if high[i] >= cand_val * (1.0 + deviation):
                if len(pivots) == 0 or (cand_bar - last_bar) >= depth:
                    pivots.append((cand_bar, float(cand_val), False))
                    last_bar = cand_bar
                    last_val = cand_val
                direction = "up"
                cand_bar = i
                cand_val = high[i]

    if len(pivots) > 0 and (cand_bar - last_bar) >= depth:
        # direction=='up' 表示正在追踪高点，故最后一拐点为高点
        pivots.append((cand_bar, float(cand_val), direction == "up"))
    return pivots


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


def calc_atr(high, low, close, period=14):
    """
    Average True Range (ATR).
    True Range = max(H-L, |H-PrevClose|, |L-PrevClose|).
    ATR = EMA(TR, period). Returns same shape as close.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = np.maximum(tr1.values, np.maximum(tr2.values, tr3.values))
    tr = pd.DataFrame(tr, index=close.index, columns=close.columns) if isinstance(close, pd.DataFrame) else pd.Series(tr, index=close.index)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


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
