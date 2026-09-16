"""
Signal 1: order-book imbalance on the Event Contract's own CLOB.

This is the most native signal (comes directly from DreamDEX's market) but it
CANNOT be backtested until either:
  (a) DreamDEX exposes historical Event Contract book data via API, or
  (b) you start recording it live yourself (see live/recorder.py, TODO), or
  (c) you use the synthetic proxy below, clearly labeled as such in any report.

Do not present synthetic-proxy backtest results as if they used real book data.
"""
import math
import numpy as np
import pandas as pd


def compute_imbalance(bid_volume: float, ask_volume: float) -> float:
    """
    Real signal, computed over top-N levels of the live Event Contract book.
    Positive -> more buy pressure on Up -> bullish tilt.
    Wire this to live/client.py once Event Contract book endpoints are confirmed.
    """
    total = bid_volume + ask_volume
    if total == 0:
        return 0.0
    return (bid_volume - ask_volume) / total


def synthetic_market_prob_up(current_price: float, open_price: float,
                              time_remaining_frac: float, vol_scale: float = 0.01,
                              sensitivity: float = 6.0) -> float:
    """
    SYNTHETIC PROXY ONLY — approximates what the Event Contract's Up price
    might look like, as a function of:
      - how far price has moved from open (diff_pct)
      - how much time is left in the window (less time -> proxy should react
        more strongly to the same diff_pct, mimicking convergence to 0/1
        as settlement approaches)
      - vol_scale: typical fractional move size for this asset/window, used
        to normalize diff_pct before the logistic squash

    Returns a probability in (0.02, 0.98). This is NOT real market data —
    label any backtest that uses it as "synthetic proxy" in your report.
    """
    if open_price == 0:
        return 0.5

    diff_pct = (current_price - open_price) / open_price
    time_decay_boost = 1.0 + (1.0 - max(0.0, min(1.0, time_remaining_frac))) * 1.5
    z = sensitivity * time_decay_boost * (diff_pct / max(vol_scale, 1e-6))

    prob = 1.0 / (1.0 + math.exp(-z))
    return float(np.clip(prob, 0.02, 0.98))


def compute_imbalance_from_book_snapshot(book: dict, top_n: int = 5) -> float:
    """
    Convenience wrapper once you have a real book snapshot, e.g.:
      book = {"bids": [(price, size), ...], "asks": [(price, size), ...]}
    TODO: confirm actual DreamDEX Event Contract book response shape and
    adjust field access below (this assumes a Level-2 style list of tuples).
    """
    bids = book.get("bids", [])[:top_n]
    asks = book.get("asks", [])[:top_n]
    bid_volume = sum(size for _, size in bids)
    ask_volume = sum(size for _, size in asks)
    return compute_imbalance(bid_volume, ask_volume)
