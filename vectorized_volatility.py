"""
Vectorized replacement for the per-window ticks_in_window() + compute_atr_regime()
pattern, which was scanning the full df1m dataframe on every window (the same
class of bug as the original momentum issue, just in data/windows.py this
time). This computes true range and tick counts for ALL windows in one
groupby pass instead of ~8,641 individual full-dataframe filters.
"""
import numpy as np
import pandas as pd


def compute_window_stats(df1m: pd.DataFrame, windows: pd.DataFrame, freq: str,
                          low_threshold: float = 0.0006, high_threshold: float = 0.006,
                          mode: str = "hard") -> pd.DataFrame:
    """
    Returns a DataFrame aligned with `windows` (same index), with columns:
      - tick_count: number of 1m ticks that fall in this window
      - vol_regime: "SIT_OUT" | "REDUCE" | "TRADE"

    Uses a global (continuous) true-range calculation, then groups by which
    window each tick belongs to -- one pass over df1m instead of one pass
    PER window.
    """
    df = df1m.sort_values("open_time").copy()

    # Continuous true range across the whole series (prev_close from the
    # actual previous tick globally, not reset per window -- slightly more
    # correct than the original per-window version, and vastly cheaper).
    prev_close = df["close"].shift(1).fillna(df["close"])
    tr = np.maximum.reduce([
        (df["high"] - df["low"]).to_numpy(),
        (df["high"] - prev_close).abs().to_numpy(),
        (df["low"] - prev_close).abs().to_numpy(),
    ])
    df["tr"] = tr

    # Assign each tick to its window bucket (window start time).
    df["window_start"] = df["open_time"].dt.floor(freq)

    grouped = df.groupby("window_start").agg(
        tick_count=("tr", "size"),
        atr=("tr", "mean"),
    ).reset_index()

    result = windows[["open_time", "open"]].merge(
        grouped, left_on="open_time", right_on="window_start", how="left"
    )
    result["tick_count"] = result["tick_count"].fillna(0).astype(int)
    result["atr"] = result["atr"].fillna(0.0)

    atr_frac = result["atr"] / result["open"].replace(0, np.nan)
    atr_frac = atr_frac.fillna(0.0)

    conditions = [
        atr_frac < low_threshold,
        atr_frac > high_threshold,
    ]
    if mode == "soft":
        choices = ["SIT_OUT", "REDUCE"]
    else:
        choices = ["SIT_OUT", "SIT_OUT"]

    result["vol_regime"] = np.select(conditions, choices, default="TRADE")
    result.index = windows.index
    return result[["tick_count", "vol_regime"]]
