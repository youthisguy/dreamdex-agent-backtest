"""
Runs momentum, mean_reversion, and confluence across a grid of spans and
thresholds, and reports ONLY combinations that clear a real statistical bar
-- protects you from picking a threshold that just happened to look good
(p-hacking) and reporting a fake edge to judges.

Usage (from project root):
    python3 backtest/grid_search.py --csv data/raw/btcusdt_1m.csv --freq 15min
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.simulate import run_backtest

MIN_TRADES = 100          # don't trust a win rate below this sample size
SIGNIFICANCE_P = 0.05     # two-sided binomial test vs 0.5


def binomial_test_vs_half(n_wins: int, n_trades: int) -> float:
    """Two-sided p-value for observed win rate vs. a fair coin, normal approx."""
    if n_trades == 0:
        return 1.0
    p_hat = n_wins / n_trades
    se = np.sqrt(0.25 / n_trades)
    z = (p_hat - 0.5) / se
    # two-sided p-value from z, using the error function (no scipy dependency)
    from math import erf
    p_value = 2 * (1 - 0.5 * (1 + erf(abs(z) / np.sqrt(2))))
    return p_value


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/btcusdt_1m.csv")
    parser.add_argument("--freq", default="15min")
    args = parser.parse_args()

    df1m = pd.read_csv(args.csv, parse_dates=["open_time", "close_time"])

    strategies = ["momentum", "mean_reversion", "confluence"]
    span_options = [(3, 12), (5, 20), (8, 30)]
    threshold_options = [0.00025, 0.0005, 0.001, 0.0015]

    results = []

    for strategy in strategies:
        for fast, slow in span_options:
            for threshold in threshold_options:
                all_windows, trades = run_backtest(
                    df1m, freq=args.freq, entry_threshold=threshold,
                    momentum_fast=fast, momentum_slow=slow, strategy=strategy,
                )
                n_trades = len(trades)
                if n_trades == 0:
                    continue
                n_wins = int(trades["won"].sum())
                win_rate = n_wins / n_trades
                p_value = binomial_test_vs_half(n_wins, n_trades)

                results.append({
                    "strategy": strategy, "fast": fast, "slow": slow,
                    "threshold": threshold, "n_trades": n_trades,
                    "win_rate": win_rate, "total_pnl": trades["pnl"].sum(),
                    "p_value": p_value,
                    "significant": (n_trades >= MIN_TRADES and p_value < SIGNIFICANCE_P),
                })

    results_df = pd.DataFrame(results).sort_values("p_value")

    out_path = Path("backtest/out/grid_search.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    print(f"Ran {len(results_df)} combinations. Saved full results -> {out_path}")
    print()
    sig = results_df[results_df["significant"]]
    if len(sig) == 0:
        print(f"No combination cleared n>={MIN_TRADES} trades AND p<{SIGNIFICANCE_P} vs a coin flip.")
        print("Honest conclusion: none of these signals show a statistically real edge yet.")
        print("\nTop 10 by p-value anyway (for inspection, NOT for reporting as a finding):")
        print(results_df.head(10).to_string(index=False))
    else:
        print(f"{len(sig)} combination(s) cleared the significance bar:")
        print(sig.sort_values("win_rate", ascending=False).to_string(index=False))
        print("\nReminder: this is still vs. a FLAT 0.5 synthetic market, not the real")
        print("DreamDEX CLOB price. Treat as 'beats a coin flip', not 'beats the real market'.")


if __name__ == "__main__":
    main()
