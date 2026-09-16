"""
Walk-forward validation.

The grid search so far picked the best-looking config using the SAME data it
was scored on -- classic in-sample overfitting risk. This script:

  1. Splits the dataset chronologically into TRAIN (first ~65%) and
     TEST (last ~35%), never randomly -- a time-series must be split in time
     order, or you leak future information into "training."
  2. Runs the same 36-combination grid on TRAIN only, applies a
     Bonferroni-corrected significance bar, and picks the best surviving
     config.
  3. Locks that config and re-evaluates it ONLY on TEST -- data it never
     influenced the choice of threshold/spans with.
  4. Reports both, side by side, so you can see whether the edge holds up
     out-of-sample or was a product of overfitting to 90 days of noise.

Usage:
    python3 backtest/walk_forward.py --csv data/raw/btcusdt_1m.csv --freq 15min
"""
import sys
from math import erf
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.simulate import run_backtest

MIN_TRADES = 100
TRAIN_FRACTION = 0.65


def binomial_p_value(n_wins: int, n_trades: int) -> float:
    if n_trades == 0:
        return 1.0
    p_hat = n_wins / n_trades
    se = np.sqrt(0.25 / n_trades)
    z = (p_hat - 0.5) / se
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / np.sqrt(2))))


def summarize(trades: pd.DataFrame) -> dict:
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "win_rate": None, "total_pnl": 0.0, "p_value": 1.0}
    wins = int(trades["won"].sum())
    win_rate = wins / n
    return {
        "n_trades": n,
        "win_rate": win_rate,
        "total_pnl": float(trades["pnl"].sum()),
        "p_value": binomial_p_value(wins, n),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/btcusdt_1m.csv")
    parser.add_argument("--freq", default="15min")
    args = parser.parse_args()

    df1m = pd.read_csv(args.csv, parse_dates=["open_time", "close_time"])
    df1m = df1m.sort_values("open_time").reset_index(drop=True)

    start, end = df1m["open_time"].iloc[0], df1m["open_time"].iloc[-1]
    split_time = start + (end - start) * TRAIN_FRACTION
    print(f"Data range: {start} -> {end}")
    print(f"Train: {start} -> {split_time}")
    print(f"Test:  {split_time} -> {end}")
    print()

    strategies = ["momentum", "mean_reversion", "confluence"]
    span_options = [(3, 12), (5, 20), (8, 30)]
    threshold_options = [0.00025, 0.0005, 0.001, 0.0015]

    n_combos = len(strategies) * len(span_options) * len(threshold_options)
    bonferroni_bar = 0.05 / n_combos
    print(f"Running {n_combos} combinations on TRAIN only. Bonferroni bar: p < {bonferroni_bar:.5f}")
    print()

    candidates = []

    for strategy in strategies:
        for fast, slow in span_options:
            for threshold in threshold_options:
                all_windows, trades = run_backtest(
                    df1m, freq=args.freq, entry_threshold=threshold,
                    momentum_fast=fast, momentum_slow=slow, strategy=strategy,
                )
                if len(trades) == 0:
                    continue

                trades = trades.copy()
                trades["time"] = pd.to_datetime(trades["time"])
                train_trades = trades[trades["time"] < split_time]
                test_trades = trades[trades["time"] >= split_time]

                train_stats = summarize(train_trades)
                if train_stats["n_trades"] < MIN_TRADES:
                    continue
                if train_stats["p_value"] >= bonferroni_bar:
                    continue

                test_stats = summarize(test_trades)

                candidates.append({
                    "strategy": strategy, "fast": fast, "slow": slow, "threshold": threshold,
                    "train_n": train_stats["n_trades"], "train_win_rate": train_stats["win_rate"],
                    "train_p": train_stats["p_value"],
                    "test_n": test_stats["n_trades"], "test_win_rate": test_stats["win_rate"],
                    "test_p": test_stats["p_value"], "test_pnl": test_stats["total_pnl"],
                })

    if not candidates:
        print("No config survived the Bonferroni-corrected bar on TRAIN.")
        print("Honest conclusion: no robust edge found even before out-of-sample testing.")
        return

    cand_df = pd.DataFrame(candidates).sort_values("train_p")
    # Only configs with a WINNING train win rate are candidates to trade on;
    # a low p-value from a losing win rate (like mean_reversion here) is
    # useful as corroborating evidence but should never be picked as "best."
    profitable = cand_df[cand_df["train_win_rate"] > 0.5]
    if len(profitable) > 0:
        cand_df = pd.concat([profitable, cand_df[cand_df["train_win_rate"] <= 0.5]])
    out_path = Path("backtest/out/walk_forward.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cand_df.to_csv(out_path, index=False)

    print(f"{len(cand_df)} config(s) survived TRAIN significance. Full table -> {out_path}")
    print()
    print(cand_df.to_string(index=False))
    print()

    best = cand_df.iloc[0]
    print("=" * 60)
    print(f"BEST TRAIN CONFIG: {best['strategy']} fast={best['fast']} slow={best['slow']} threshold={best['threshold']}")
    print(f"  TRAIN: n={best['train_n']}, win_rate={best['train_win_rate']:.2%}, p={best['train_p']:.5f}")
    if best["test_n"] < MIN_TRADES:
        print(f"  TEST:  only {best['test_n']} trades -- too few to draw a conclusion. Consider a longer date range.")
    else:
        holds = best["test_win_rate"] > 0.5 and best["test_p"] < 0.05
        print(f"  TEST:  n={best['test_n']}, win_rate={best['test_win_rate']:.2%}, p={best['test_p']:.5f}, pnl={best['test_pnl']:.2f}")
        if holds:
            print("  -> Edge HOLDS out-of-sample. This is a legitimate, defensible finding.")
        else:
            print("  -> Edge does NOT clearly hold out-of-sample. Likely overfit to the train period.")
            print("     Report this honestly -- it's still a real, useful result for your submission.")
    print("=" * 60)


if __name__ == "__main__":
    main()