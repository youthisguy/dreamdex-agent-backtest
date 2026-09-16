"""
Runs the combined signal against historical windows, no real trades.

MODIFIED to support a `strategy` switch: "momentum" (existing, fixed
version), "mean_reversion" (new hypothesis), or "confluence" (momentum
gated by long-horizon trend agreement). All three write to the same
Decision/record shape so grid_search.py and the dashboard don't need to
change.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.windows import build_windows, ticks_in_window
from signals.momentum import compute_continuous_emas, momentum_at, compute_long_emas, confluence_signal
from signals.mean_reversion import compute_continuous_return_stats, mean_reversion_at
from signals.volatility import compute_atr_regime
from signals.imbalance import synthetic_market_prob_up
from signals.combine import decide


def run_backtest(df1m: pd.DataFrame, freq: str = "15min", entry_threshold: float = 0.06,
                  w1: float = 0.5, w2: float = 0.5, stake: float = 10.0,
                  use_synthetic_market: bool = False, vol_low: float = 0.0006,
                  vol_high: float = 0.006, momentum_fast: int = 5, momentum_slow: int = 20,
                  strategy: str = "momentum", mr_lookback: int = 10, mr_vol_lookback: int = 20,
                  long_fast: int = 60, long_slow: int = 240):
    """
    strategy: "momentum" | "mean_reversion" | "confluence"
    Returns (all_windows, trades) DataFrames, same shape regardless of strategy.
    """
    windows = build_windows(df1m, freq=freq)

    # Precompute whatever continuous state the chosen strategy needs, ONCE,
    # outside the loop -- this is the pattern that fixed the original bug.
    df1m_short = compute_continuous_emas(df1m, fast=momentum_fast, slow=momentum_slow)

    if strategy == "mean_reversion":
        df1m_mr = compute_continuous_return_stats(df1m, lookback=mr_vol_lookback)
    elif strategy == "confluence":
        df1m_long = compute_long_emas(df1m, fast=long_fast, slow=long_slow)

    records = []

    for _, window in windows.iterrows():
        ticks = ticks_in_window(df1m, window.open_time, freq=freq)
        if len(ticks) < 3:
            continue

        if strategy == "mean_reversion":
            signal_value = mean_reversion_at(df1m_mr, window.open_time, window.open,
                                              move_lookback_minutes=mr_lookback)
        elif strategy == "confluence":
            signal_value = confluence_signal(df1m_short, df1m_long, window.open_time, window.open)
        else:  # "momentum"
            signal_value = momentum_at(df1m_short, window.open_time, window.open)

        vol_regime = compute_atr_regime(ticks, window.open, low_threshold=vol_low,
                                         high_threshold=vol_high)

        # Signal 1 stub: no real historical Event Contract book yet.
        imbalance = 0.0

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