"""
Scratch sanity check for book_reconstruction.py -- run this against ONE
market before wiring anything into backtest/simulate.py.

Usage:
    python3 check_book_reconstruction.py 0x3962d639F220f40d2a96fC5E5E217fb387dfA8c1

Right now this will raise NotImplementedError, because fetch_order_events()
in data/book_reconstruction.py is still a stub -- see the TODO in that file.
It needs a real indexer or RPC call wired in first (that's the "send me a
sample response" step), and this script only becomes runnable after that.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.book_reconstruction import MarketReplay, market_prob_up_at, fetch_order_events


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 check_book_reconstruction.py <market_id>")
        sys.exit(1)

    market_id = sys.argv[1]
    events = fetch_order_events(market_id)
    if not events:
        print(f"No events returned for {market_id} -- check the market_id or the fetch implementation.")
        return

    replay = MarketReplay(events)
    checkpoints = [
        events[0]["block"],
        events[len(events) // 4]["block"],
        events[len(events) // 2]["block"],
        events[3 * len(events) // 4]["block"],
        events[-1]["block"],
    ]

    print(f"{market_id}: {len(events)} events, blocks {events[0]['block']} -> {events[-1]['block']}")
    print()
    print(f"{'block':>12}  {'market_prob_up':>15}")
    for b in checkpoints:
        p = market_prob_up_at(replay, b)
        p_str = f"{p:.4f}" if p is not None else "None (empty book)"
        print(f"{b:>12}  {p_str:>15}")

    print()
    print("Sanity check: does this look like a plausible price path?")
    print("  - Early blocks near 0.5 (market just opened, little info yet)")
    print("  - Later blocks trending toward 0 or 1 as the market approaches resolution")
    print("  - No wild jumps that look like a schema/field-mapping bug rather than real price action")


if __name__ == "__main__":
    main()
