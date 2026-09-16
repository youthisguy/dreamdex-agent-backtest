"""
Vectorized replacements for the per-window momentum_at / mean_reversion_at /
confluence_signal lookups, which were each doing a full dataframe scan
(df[df.open_time <= timestamp].tail(1)) on every call -- O(n) per window,
called ~300,000 times across a full grid search, which is why it looked
"stuck". pd.merge_asof does the same "most recent row at/before timestamp"
lookup for ALL windows at once in a single vectorized pass.

Usage: call these ONCE per backtest run (not per window) to get a Series
aligned with your `windows` dataframe, then just index into it in the loop.
"""
import numpy as np
import pandas as pd


def momentum_series_for_windows(df1m_with_emas: pd.DataFrame, windows: pd.DataFrame,
                                 cap: float = 0.05) -> pd.Series:
    """
    Vectorized version of calling momentum_at() once per window.
    df1m_with_emas: output of compute_continuous_emas(), sorted by open_time.
    windows: your windows dataframe (from build_windows), needs open_time, open.
    Returns a Series aligned with windows.index.
    """
    left = windows[["open_time", "open"]].sort_values("open_time")
    right = df1m_with_emas[["open_time", "ema_fast", "ema_slow"]].sort_values("open_time")

    merged = pd.merge_asof(left, right, on="open_time", direction="backward")
    diff_norm = (merged["ema_fast"] - merged["ema_slow"]) / merged["open"].replace(0, np.nan)
    signal = np.sign(diff_norm) * diff_norm.abs().clip(upper=cap)
    signal = signal.fillna(0.0)
    # restore original windows order/index
    signal.index = left.index
    return signal.reindex(windows.index)


def mean_reversion_series_for_windows(df1m_with_stats: pd.DataFrame, windows: pd.DataFrame,
                                       move_lookback_minutes: int = 10, cap: float = 0.05,
                                       z_scale: float = 0.01) -> pd.Series:
    """
    Vectorized version of mean_reversion_at(). Precomputes the lookback-minute
    return ending at each 1m tick, then asof-joins onto window open times.
    """
    df = df1m_with_stats.sort_values("open_time").copy()
    df["move_lookback_ret"] = df["close"].pct_change(periods=move_lookback_minutes)

    left = windows[["open_time", "open"]].sort_values("open_time")
    right = df[["open_time", "move_lookback_ret", "roll_std"]]

    merged = pd.merge_asof(left, right, on="open_time", direction="backward")
    z = merged["move_lookback_ret"] / (merged["roll_std"] * np.sqrt(move_lookback_minutes))
    signal = -np.sign(z) * (z.abs() * z_scale).clip(upper=cap)
    signal = signal.fillna(0.0)
    signal.index = left.index
    return signal.reindex(windows.index)


def confluence_series_for_windows(df1m_short_emas: pd.DataFrame, df1m_long_emas: pd.DataFrame,
                                   windows: pd.DataFrame, cap: float = 0.05) -> pd.Series:
    """
    Vectorized confluence: short momentum gated by long-trend sign agreement.
    """
    short_signal = momentum_series_for_windows(df1m_short_emas, windows, cap=cap)

    left = windows[["open_time"]].sort_values("open_time")
    right = df1m_long_emas[["open_time", "ema_fast_long", "ema_slow_long"]].sort_values("open_time")
    merged = pd.merge_asof(left, right, on="open_time", direction="backward")
    long_diff = merged["ema_fast_long"] - merged["ema_slow_long"]
    long_sign = np.sign(long_diff).fillna(0.0)
    long_sign.index = left.index
    long_sign = long_sign.reindex(windows.index)

    short_sign = np.sign(short_signal)
    agree = (long_sign != 0) & (short_sign != 0) & (long_sign == short_sign)
    return short_signal.where(agree, 0.0)
