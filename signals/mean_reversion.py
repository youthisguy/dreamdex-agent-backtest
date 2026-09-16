"""
Signal alternative: mean-reversion instead of momentum.

Hypothesis: on a very short horizon (15-min windows), a sharp move is more
likely to partially revert than to continue -- the opposite bet from
momentum. Implemented as a z-score of the recent move relative to recent
volatility: the more "extreme" the move (in standard-deviation terms), the
stronger the reversion signal, capped the same way momentum is.
"""
import numpy as np
import pandas as pd


def compute_continuous_return_stats(df1m: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Call once on the full 1m dataframe. Adds:
      - 'ret_1m': 1-minute pct return
      - 'roll_std': rolling std of ret_1m over `lookback` minutes (recent vol)
    """
    df1m = df1m.copy()
    df1m["ret_1m"] = df1m["close"].pct_change()
    df1m["roll_std"] = df1m["ret_1m"].rolling(lookback).std()
    return df1m


def mean_reversion_at(df1m_with_stats: pd.DataFrame, timestamp, open_price: float,
                       move_lookback_minutes: int = 10, cap: float = 0.05,
                       z_scale: float = 0.01) -> float:
    """
    Reads price change over the last `move_lookback_minutes` ending at/before
    `timestamp`, normalizes it by recent realized vol (z-score), and returns
    a signal that is the OPPOSITE sign of the recent move (reversion bet).

    Positive output -> bet UP (i.e. recent move was down and we expect reversion up).
    Same [-cap, cap] contract as momentum_signal so it plugs into combine.decide() unchanged.
    """
    hist = df1m_with_stats[df1m_with_stats["open_time"] <= timestamp]
    if len(hist) < move_lookback_minutes + 1 or open_price == 0:
        return 0.0

    recent = hist.tail(move_lookback_minutes + 1)
    move = (recent["close"].iloc[-1] - recent["close"].iloc[0]) / recent["close"].iloc[0]

    roll_std = hist["roll_std"].iloc[-1]
    if not roll_std or np.isnan(roll_std) or roll_std == 0:
        return 0.0

    z = move / (roll_std * np.sqrt(move_lookback_minutes))
    reversion_signal = -np.sign(z) * min(abs(z) * z_scale, cap)
    return float(reversion_signal)
