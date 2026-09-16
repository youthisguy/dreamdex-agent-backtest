"""
Reconstructs Event-Contract-style windows (15min or 1h) from 1-minute klines,
producing ground truth (did Up or Down actually win) for every historical window.
"""
import pandas as pd


def load_klines(path):
    df = pd.read_csv(path, parse_dates=["open_time", "close_time"])
    return df.sort_values("open_time").reset_index(drop=True)


def build_windows(df1m, freq="15min"):
    """
    Resample 1m klines into windows matching DreamDEX's Event Contract cadence.
    freq: "15min" or "1h"
    """
    df = df1m.set_index("open_time")
    windows = df.resample(freq).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna().reset_index()
    windows["outcome_up"] = windows["close"] >= windows["open"]
    return windows


def ticks_in_window(df1m, window_open_time, freq="15min"):
    """All 1m ticks belonging to a given window, for computing signals inside it."""
    delta = pd.Timedelta(freq)
    mask = (df1m.open_time >= window_open_time) & (df1m.open_time < window_open_time + delta)
    return df1m.loc[mask]
