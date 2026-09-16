"""
APPEND THIS FUNCTION to your existing signals/momentum.py (don't replace the
file -- just add this alongside compute_continuous_emas, momentum_at, and
compute_raw_momentum, which you already have).

Confluence: only trust the short-term momentum signal when a longer-horizon
trend agrees with it. If they disagree, sit out -- the idea is that a 15-min
move against the prevailing longer trend is more likely noise.
"""
import numpy as np


def compute_long_emas(df1m, fast: int = 60, slow: int = 240):
    """
    Same idea as compute_continuous_emas but with much longer spans (in
    minutes) to represent a longer-horizon trend -- default 60/240 min
    (~1h/4h). Call this once, same pattern as compute_continuous_emas.
    """
    df1m = df1m.copy()
    df1m["ema_fast_long"] = df1m["close"].ewm(span=fast, adjust=False).mean()
    df1m["ema_slow_long"] = df1m["close"].ewm(span=slow, adjust=False).mean()
    return df1m


def confluence_signal(df1m_short_emas, df1m_long_emas, timestamp, open_price: float,
                       cap: float = 0.05) -> float:
    """
    Returns the short-term momentum_signal value ONLY if the long-horizon
    trend agrees in sign; otherwise returns 0.0 (sit out).

    df1m_short_emas: output of compute_continuous_emas() (has ema_fast/ema_slow)
    df1m_long_emas: output of compute_long_emas() (has ema_fast_long/ema_slow_long)
    Both must share the same 'open_time' column to align on.
    """
    from signals.momentum import momentum_at  # short-term signal, already have this

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
