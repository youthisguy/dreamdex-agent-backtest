"""
Runs the combined signal against historical windows, no real trades.

FULLY VECTORIZED: both the signal computation (momentum/mean_reversion/
confluence) AND the tick-count/ATR-regime computation now happen once, for
all windows at once, via merge_asof / groupby (see vectorized_signals.py and
vectorized_volatility.py). The per-window loop below does ONLY cheap index
lookups -- no dataframe scanning happens inside the loop anymore. This
replaces the old ticks_in_window() calls from data/windows.py, which were
scanning the full ~130k-row dataframe once per window (8,641 times per run)
and were the real cause of the "stuck" grid search.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.windows import build_windows
from signals.momentum import compute_continuous_emas, compute_long_emas
from signals.mean_reversion import compute_continuous_return_stats
from signals.imbalance import synthetic_market_prob_up
from signals.combine import decide
from vectorized_signals import (
    momentum_series_for_windows,
    mean_reversion_series_for_windows,
    confluence_series_for_windows,
)
from vectorized_volatility import compute_window_stats


def run_backtest(df1m: pd.DataFrame, freq: str = "15min", entry_threshold: float = 0.06,
                  w1: float = 0.5, w2: float = 0.5, stake: float = 10.0,
                  use_synthetic_market: bool = False, vol_low: float = 0.0006,
                  vol_high: float = 0.006, momentum_fast: int = 5, momentum_slow: int = 20,
                  strategy: str = "momentum", mr_lookback: int = 10, mr_vol_lookback: int = 20,
                  long_fast: int = 60, long_slow: int = 240):
    windows = build_windows(df1m, freq=freq).reset_index(drop=True)

    df1m_short = compute_continuous_emas(df1m, fast=momentum_fast, slow=momentum_slow)

    if strategy == "mean_reversion":
        df1m_mr = compute_continuous_return_stats(df1m, lookback=mr_vol_lookback)
        signal_series = mean_reversion_series_for_windows(df1m_mr, windows, move_lookback_minutes=mr_lookback)
    elif strategy == "confluence":
        df1m_long = compute_long_emas(df1m, fast=long_fast, slow=long_slow)
        signal_series = confluence_series_for_windows(df1m_short, df1m_long, windows)
    else:
        signal_series = momentum_series_for_windows(df1m_short, windows)

    # Vectorized: tick counts + volatility regime for ALL windows in one pass.
    window_stats = compute_window_stats(df1m, windows, freq=freq, low_threshold=vol_low,
                                         high_threshold=vol_high)

    records = []

    for i, window in windows.iterrows():
        stats = window_stats.loc[i]
        if stats["tick_count"] < 3:
            continue

        signal_value = signal_series.loc[i]
        vol_regime = stats["vol_regime"]
        imbalance = 0.0  # Signal 1 stub: no real historical Event Contract book yet.

        if use_synthetic_market:
            market_prob_up = synthetic_market_prob_up(
                current_price=window.close, open_price=window.open,
                time_remaining_frac=0.5,
            )
        else:
            market_prob_up = 0.5

        decision = decide(imbalance, signal_value, vol_regime, market_prob_up,
                           w1=w1, w2=w2, entry_threshold=entry_threshold)

        record = {
            "time": window.open_time,
            "open": window.open,
            "close": window.close,
            "outcome_up": window.outcome_up,
            "strategy": strategy,
            **decision.to_dict(),
        }

        if decision.side is not None:
            won = (decision.side == "UP") == window.outcome_up
            entry_price = market_prob_up if decision.side == "UP" else (1 - market_prob_up)
            entry_price = max(entry_price, 0.01)
            payout = stake / entry_price if won else 0.0
            pnl = payout - stake
            record.update({"won": won, "stake": stake, "pnl": pnl})
        else:
            record.update({"won": None, "stake": 0.0, "pnl": 0.0})

        records.append(record)

    all_windows = pd.DataFrame(records)
    trades = all_windows[all_windows["side"].notna()].reset_index(drop=True)
    return all_windows, trades


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a backtest for a given strategy")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--freq", default="15min")
    parser.add_argument("--entry-threshold", type=float, default=0.0005)
    parser.add_argument("--stake", type=float, default=10.0)
    parser.add_argument("--strategy", default="momentum", choices=["momentum", "mean_reversion", "confluence"])
    args = parser.parse_args()

    df1m = pd.read_csv(args.csv, parse_dates=["open_time", "close_time"])
    all_windows, trades = run_backtest(df1m, freq=args.freq, entry_threshold=args.entry_threshold,
                                        stake=args.stake, strategy=args.strategy)

    print(f"Strategy: {args.strategy}")
    print(f"Windows evaluated: {len(all_windows)}")
    print(f"Trades taken: {len(trades)}")
    if len(trades):
        print(f"Win rate: {trades['won'].mean():.2%}")
        print(f"Total PnL: {trades['pnl'].sum():.2f}")