"""
Signal 2: short-term momentum using continuous EMA state.

compute EMA(fast)/EMA(slow) ONCE across the full continuous price
series (so they carry real trend state), then just read off the values at
each window's boundary. This is also the more realistic version of what a
live agent would do -- it maintains rolling EMA state continuously, it
doesn't reset context every 15 minutes.
"""
import numpy as np
import pandas as pd


def compute_continuous_emas(df1m: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.DataFrame:
    """
    Call this ONCE on your full 1m dataframe (must have a 'close' column,
    sorted by time). Returns df1m with 'ema_fast' and 'ema_slow' columns
    added, computed continuously across the whole series.
    """
    df1m = df1m.copy()
    df1m["ema_fast"] = df1m["close"].ewm(span=fast, adjust=False).mean()
    df1m["ema_slow"] = df1m["close"].ewm(span=slow, adjust=False).mean()
    return df1m


def momentum_at(df1m_with_emas: pd.DataFrame, timestamp, open_price: float,
                 cap: float = 0.05) -> float:
    """
    Reads the continuous EMA state at (or just before) `timestamp` -- e.g. the
    last tick before a window opens, or the current live tick -- and returns
    the normalized, capped momentum signal.

    df1m_with_emas: output of compute_continuous_emas(), with a datetime
                     column named 'open_time' to match against `timestamp`.
    """
    row = df1m_with_emas[df1m_with_emas["open_time"] <= timestamp].tail(1)
    if row.empty or open_price == 0:
        return 0.0

    ema_fast = row["ema_fast"].iloc[0]
    ema_slow = row["ema_slow"].iloc[0]
    diff_norm = (ema_fast - ema_slow) / open_price
    return float(np.sign(diff_norm) * min(abs(diff_norm), cap))


# --- Backward-compatible fallback, kept for the flat-open-vs-current check ---
def compute_raw_momentum(current_price: float, open_price: float) -> float:
    """Simple (current - open) / open. Unaffected by the EMA bug -- fine as a fallback."""
    if open_price == 0:
        return 0.0
    return (current_price - open_price) / open_price


# --- Confluence: gate short-term momentum on longer-horizon trend agreement ---

def compute_long_emas(df1m: pd.DataFrame, fast: int = 60, slow: int = 240) -> pd.DataFrame:
    """
    Same idea as compute_continuous_emas but with much longer spans (in
    minutes) to represent a longer-horizon trend -- default 60/240 min
    (~1h/4h). Call this once, same pattern as compute_continuous_emas.
    """
    df1m = df1m.copy()
    df1m["ema_fast_long"] = df1m["close"].ewm(span=fast, adjust=False).mean()
    df1m["ema_slow_long"] = df1m["close"].ewm(span=slow, adjust=False).mean()
    return df1m


def confluence_signal(df1m_short_emas: pd.DataFrame, df1m_long_emas: pd.DataFrame,
                       timestamp, open_price: float, cap: float = 0.05) -> float:
    """
    Returns the short-term momentum_signal value ONLY if the long-horizon
    trend agrees in sign; otherwise returns 0.0 (sit out).

    df1m_short_emas: output of compute_continuous_emas() (has ema_fast/ema_slow)
    df1m_long_emas: output of compute_long_emas() (has ema_fast_long/ema_slow_long)
    Both must share the same 'open_time' column to align on.
    """
    short_signal = momentum_at(df1m_short_emas, timestamp, open_price, cap=cap)

    long_row = df1m_long_emas[df1m_long_emas["open_time"] <= timestamp].tail(1)
    if long_row.empty:
        return 0.0

    long_diff = long_row["ema_fast_long"].iloc[0] - long_row["ema_slow_long"].iloc[0]
    long_sign = np.sign(long_diff)
    short_sign = np.sign(short_signal)

    if long_sign == 0 or short_sign == 0 or long_sign != short_sign:
        return 0.0  # disagreement (or no signal) -> sit out

    return short_signal