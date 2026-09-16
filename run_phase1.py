"""
One-command Phase 1 pipeline: fetch Binance history (if not cached) -> run
backtest with the synthetic market-price proxy -> generate a report.

Usage:
    python run_phase1.py --symbol BTCUSDT --days 90 --freq 15min
"""
import argparse
from pathlib import Path

from data.fetch_binance import fetch_klines
from backtest.report import generate_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--freq", default="15min")
    parser.add_argument("--entry-threshold", type=float, default=0.06)
    parser.add_argument("--stake", type=float, default=10.0)
    parser.add_argument("--vol-low", type=float, default=0.0006)
    parser.add_argument("--vol-high", type=float, default=0.006)
    parser.add_argument("--refresh", action="store_true", help="Re-fetch even if a cached CSV exists")
    args = parser.parse_args()

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / f"{args.symbol.lower()}_1m.csv"

    if args.refresh or not csv_path.exists():
        print(f"Fetching {args.days}d of 1m klines for {args.symbol}...")
        import time
        df = fetch_klines(args.symbol, "1m", start_ms=int((time.time() - args.days * 86400) * 1000))
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} rows -> {csv_path}")
    else:
        print(f"Using cached data -> {csv_path} (pass --refresh to re-fetch)")

    generate_report(str(csv_path), out_dir="backtest/out", freq=args.freq,
                     entry_threshold=args.entry_threshold, stake=args.stake,
                     vol_low=args.vol_low, vol_high=args.vol_high)


if __name__ == "__main__":
    main()
