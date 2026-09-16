"""
Signal 3: volatility regime filter. Decides whether the window should be
traded at all, using a simple ATR (average true range) over the ticks
observed so far in the window.

Regimes:
  SIT_OUT  -> ATR below low_threshold (chop, no edge) or above high_threshold
              (too noisy / gap risk near settlement)
  REDUCE   -> ATR above high_threshold but caller wants a soft mode instead
              of a hard skip (position sizing lever, unused in v1 backtest)
  TRADE    -> normal regime
"""
import numpy as np
import pandas as pd


def average_true_range(ticks: pd.DataFrame) -> float:
    """
    ATR computed from high/low/close of the ticks in-window.
    For 1m klines this approximates true range as high-low plus the gap
    from the previous close.
    """
    if len(ticks) < 2:
        return 0.0

    high = ticks["high"].to_numpy()
    low = ticks["low"].to_numpy()
    close = ticks["close"].to_numpy()
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    return float(np.mean(tr))


def compute_atr_regime(ticks: pd.DataFrame, open_price: float,
                        low_threshold: float = 0.0006, high_threshold: float = 0.006,
                        mode: str = "hard") -> str:
    """
    Thresholds are expressed as ATR / open_price (fractional), so they scale
    across BTC/ETH price levels. Defaults are a starting point — tune in backtest.

    mode="hard": both extremes return SIT_OUT
    mode="soft": high extreme returns REDUCE instead of SIT_OUT
    """
    if open_price == 0:
        return "SIT_OUT"

    atr = average_true_range(ticks)
    atr_frac = atr / open_price

    if atr_frac < low_threshold:
        return "SIT_OUT"
    if atr_frac > high_threshold:
        return "REDUCE" if mode == "soft" else "SIT_OUT"
    return "TRADE"
