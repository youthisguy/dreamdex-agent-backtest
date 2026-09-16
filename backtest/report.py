"""
Generates a summary printout + cumulative PnL chart (PNG) + CSV export from a
backtest run. Meant to be run standalone against a fetched CSV of klines, or
imported and called from a notebook / phase1 runner.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.simulate import run_backtest
from backtest.metrics import compute_metrics, breakdown_by_hour, breakdown_by_regime


def generate_report(csv_path: str, out_dir: str = "backtest/out", freq: str = "15min",
                     entry_threshold: float = 0.06, stake: float = 10.0,
                     vol_low: float = 0.0006, vol_high: float = 0.006):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df1m = pd.read_csv(csv_path, parse_dates=["open_time", "close_time"])
    all_windows, trades = run_backtest(df1m, freq=freq, entry_threshold=entry_threshold, stake=stake,
                                        vol_low=vol_low, vol_high=vol_high)
    metrics = compute_metrics(all_windows, trades)

    print("=" * 50)
    print(f"BACKTEST REPORT  ({csv_path}, freq={freq}, entry_threshold={entry_threshold})")
    print("NOTE: uses SYNTHETIC market-price proxy — not real Event Contract prices.")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"{k:>22}: {v}")

    all_windows.to_csv(out_path / "all_windows.csv", index=False)
    trades.to_csv(out_path / "trades.csv", index=False)

    if len(trades):
        fig, ax = plt.subplots(figsize=(10, 5))
        cumulative = trades["pnl"].cumsum()
        ax.plot(pd.to_datetime(trades["time"]), cumulative, linewidth=1.5)
        ax.set_title("Cumulative PnL (synthetic proxy backtest)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative PnL (USDso)")
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_path / "cumulative_pnl.png", dpi=150)
        plt.close(fig)
        print(f"\nSaved chart -> {out_path / 'cumulative_pnl.png'}")

        hourly = breakdown_by_hour(trades)
        hourly.to_csv(out_path / "breakdown_by_hour.csv", index=False)

    regime_counts = breakdown_by_regime(all_windows)
    regime_counts.to_csv(out_path / "breakdown_by_regime.csv", index=False)
    print(f"Saved CSVs -> {out_path}/")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a backtest report")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="backtest/out")
    parser.add_argument("--freq", default="15min")
    parser.add_argument("--entry-threshold", type=float, default=0.06)
    parser.add_argument("--stake", type=float, default=10.0)
    parser.add_argument("--vol-low", type=float, default=0.0006)
    parser.add_argument("--vol-high", type=float, default=0.006)
    args = parser.parse_args()

    generate_report(args.csv, args.out, args.freq, args.entry_threshold, args.stake,
                     args.vol_low, args.vol_high)
