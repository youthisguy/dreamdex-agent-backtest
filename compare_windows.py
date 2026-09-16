"""
Runs the SAME walk-forward methodology already in backtest/walk_forward.py
across every window DreamDEX currently supports (5min, 15min, 1h) and lines
up the results side by side.

Usage (from project root, same as walk_forward.py):
    python3 compare_windows.py --csv data/raw/btcusdt_1m.csv

Optional:
    python3 compare_windows.py --csv data/raw/btcusdt_1m.csv --freqs 5min,15min,1h
"""
import sys
from math import erf
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from backtest.simulate import run_backtest

MIN_TRADES = 100
TRAIN_FRACTION = 0.65

STRATEGIES = ["momentum", "mean_reversion", "confluence"]
SPAN_OPTIONS = [(3, 12), (5, 20), (8, 30)]
THRESHOLD_OPTIONS = [0.00025, 0.0005, 0.001, 0.0015]
N_COMBOS = len(STRATEGIES) * len(SPAN_OPTIONS) * len(THRESHOLD_OPTIONS)
BONFERRONI_BAR = 0.05 / N_COMBOS


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


def run_walk_forward_for_freq(df1m: pd.DataFrame, freq: str, split_time) -> pd.DataFrame:
    """Same TRAIN-select / TEST-confirm logic as backtest/walk_forward.py, factored
    out so it can be called once per frequency without duplicating file I/O."""
    candidates = []

    for strategy in STRATEGIES:
        for fast, slow in SPAN_OPTIONS:
            for threshold in THRESHOLD_OPTIONS:
                all_windows, trades = run_backtest(
                    df1m, freq=freq, entry_threshold=threshold,
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
                if train_stats["p_value"] >= BONFERRONI_BAR:
                    continue

                test_stats = summarize(test_trades)

                candidates.append({
                    "freq": freq, "strategy": strategy, "fast": fast, "slow": slow,
                    "threshold": threshold,
                    "train_n": train_stats["n_trades"], "train_win_rate": train_stats["win_rate"],
                    "train_p": train_stats["p_value"],
                    "test_n": test_stats["n_trades"], "test_win_rate": test_stats["win_rate"],
                    "test_p": test_stats["p_value"], "test_pnl": test_stats["total_pnl"],
                })

    if not candidates:
        return pd.DataFrame()

    cand_df = pd.DataFrame(candidates).sort_values("train_p")
    # Same rule as walk_forward.py: only a WINNING train win rate is eligible
    # to be picked as "best" -- a low p-value from a losing win rate is
    # corroborating evidence, never the headline result.
    profitable = cand_df[cand_df["train_win_rate"] > 0.5]
    if len(profitable) > 0:
        cand_df = pd.concat([profitable, cand_df[cand_df["train_win_rate"] <= 0.5]])
    return cand_df.reset_index(drop=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/btcusdt_1m.csv")
    parser.add_argument("--freqs", default="5min,15min,1h",
                         help="comma-separated list of window lengths to test")
    args = parser.parse_args()
    freqs = [f.strip() for f in args.freqs.split(",") if f.strip()]

    df1m = pd.read_csv(args.csv, parse_dates=["open_time", "close_time"])
    df1m = df1m.sort_values("open_time").reset_index(drop=True)

    start, end = df1m["open_time"].iloc[0], df1m["open_time"].iloc[-1]
    split_time = start + (end - start) * TRAIN_FRACTION
    print(f"Data range: {start} -> {end}")
    print(f"Train: {start} -> {split_time}")
    print(f"Test:  {split_time} -> {end}")
    print(f"Bonferroni bar (train significance): p < {BONFERRONI_BAR:.5f}")
    print()

    out_dir = Path("backtest/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for freq in freqs:
        print(f"=== {freq} ===")
        cand_df = run_walk_forward_for_freq(df1m, freq, split_time)

        if cand_df.empty:
            print(f"  No config survived the Bonferroni-corrected bar on TRAIN for {freq}.")
            summary_rows.append({
                "freq": freq, "verdict": "no train-significant config",
                "strategy": None, "fast": None, "slow": None, "threshold": None,
                "train_win_rate": None, "train_p": None,
                "test_n": None, "test_win_rate": None, "test_p": None, "test_pnl": None,
            })
            print()
            continue

        # Save the FULL per-frequency table -- distinct filename per freq, so
        # nothing gets clobbered the way it would calling walk_forward.py
        # directly three times in a row.
        freq_out_path = out_dir / f"walk_forward_{freq}.csv"
        cand_df.to_csv(freq_out_path, index=False)
        print(f"  {len(cand_df)} config(s) survived TRAIN significance -> {freq_out_path}")

        best = cand_df.iloc[0]
        if best["test_n"] < MIN_TRADES:
            verdict = f"only {best['test_n']} test trades -- inconclusive"
        else:
            holds = best["test_win_rate"] > 0.5 and best["test_p"] < 0.05
            verdict = "EDGE HOLDS out-of-sample" if holds else "does NOT hold out-of-sample (likely overfit)"

        print(f"  best: {best['strategy']} fast={best['fast']} slow={best['slow']} threshold={best['threshold']}")
        print(f"    train: n={best['train_n']}, win_rate={best['train_win_rate']:.2%}, p={best['train_p']:.5f}")
        if best["test_n"] > 0:
            print(f"    test:  n={best['test_n']}, win_rate={best['test_win_rate']:.2%}, "
                  f"p={best['test_p']:.5f}, pnl={best['test_pnl']:.2f}")
        print(f"    verdict: {verdict}")
        print()

        summary_rows.append({
            "freq": freq, "verdict": verdict,
            "strategy": best["strategy"], "fast": best["fast"], "slow": best["slow"],
            "threshold": best["threshold"],
            "train_win_rate": best["train_win_rate"], "train_p": best["train_p"],
            "test_n": best["test_n"], "test_win_rate": best["test_win_rate"],
            "test_p": best["test_p"], "test_pnl": best["test_pnl"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "walk_forward_window_comparison.csv"
    summary_df.to_csv(summary_path, index=False)

    print("=" * 70)
    print("WINDOW COMPARISON (best train-significant config per frequency)")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print()
    print(f"Full comparison -> {summary_path}")
    print(f"Full per-frequency candidate tables -> {out_dir}/walk_forward_<freq>.csv")
    print()
    print("Read this the same skeptical way walk_forward.py's own output should be read:")
    print("only trust a row whose TEST win rate/p-value confirms the TRAIN result --")
    print("a good train number with a bad or inconclusive test number is overfitting,")
    print("not an edge, regardless of which window it happened on.")


if __name__ == "__main__":
    main()
