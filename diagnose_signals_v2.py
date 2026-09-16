"""
Run this after applying the momentum.py + simulate.py fix, to see the real
distribution of momentum_signal values and pick an entry_threshold that
gives you a statistically meaningful number of trades (aim for at least
~30-50+ before trusting a win rate at all).

Usage (from project root):
    python3 diagnose_signals_v2.py --csv data/raw/btcusdt_1m.csv --freq 15min
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from data.windows import build_windows, ticks_in_window
from signals.momentum import compute_continuous_emas, momentum_at

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="data/raw/btcusdt_1m.csv")
parser.add_argument("--freq", default="15min")
parser.add_argument("--fast", type=int, default=5)
parser.add_argument("--slow", type=int, default=20)
args = parser.parse_args()

df1m = pd.read_csv(args.csv, parse_dates=["open_time", "close_time"])
df1m_emas = compute_continuous_emas(df1m, fast=args.fast, slow=args.slow)
windows = build_windows(df1m, freq=args.freq)

values = []
for _, window in windows.iterrows():
    ticks = ticks_in_window(df1m, window.open_time, freq=args.freq)
    if len(ticks) < 3:
        continue
    last_tick_time = ticks["open_time"].iloc[-1]
    m = momentum_at(df1m_emas, last_tick_time, window.open)
    values.append(m)

s = pd.Series(values)
abs_s = s.abs()

print(f"=== momentum_signal distribution (fast={args.fast}, slow={args.slow}) ===")
print(s.describe())
print()
print("=== |momentum_signal| percentiles (use this to pick entry_threshold) ===")
for p in [50, 75, 90, 95, 97.5, 99]:
    print(f"  p{p}: {np.percentile(abs_s, p):.6f}")
print()
print("=== implied participation rate at various thresholds ===")
for t in [0.0005, 0.001, 0.002, 0.003, 0.005, 0.01]:
    rate = (abs_s > t).mean()
    n = (abs_s > t).sum()
    print(f"  threshold={t:.4f} -> {n} windows ({rate:.2%}) would clear it")
