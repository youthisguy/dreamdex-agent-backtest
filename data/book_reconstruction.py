"""
Reconstructs an Event Contract's order book at any point in its lifetime
from a stream of order events -- so backtest/simulate.py can price against
what the market ACTUALLY quoted at decision time, instead of the flat 0.5
synthetic proxy called out in README.md and backtest/report.py.

This file is schema-agnostic on purpose: it doesn't know or guess DreamDEX's
exact indexer query shape (I don't have that -- see fetch_order_events()
below, which is the one function YOU need to fill in once you have a sample
response from docs.dreamdex.io/developers or the @somnia-chain/markets-sdk
indexer). Everything else -- the replay logic -- only needs the four event
types the docs already named: OrderPlaced, OrderCancelled, OrderFilled,
OrderReduced.

Expected event shape once you adapt fetch_order_events() to the real API,
one dict per event:
    {
        "event_type": "OrderPlaced" | "OrderCancelled" | "OrderFilled" | "OrderReduced",
        "order_id": <str, unique within the market>,
        "outcome": "YES" | "NO",           # which leg's book this order sits on
        "side": "bid" | "ask",
        "price": <float, 0..1>,            # price IS probability on this venue
        "quantity": <float>,               # size at OrderPlaced; remaining after Reduced
        "block": <int>,                    # for ordering -- use placedAtBlock /
                                            # lastUpdatedAtBlock per the docs, NOT
                                            # wall-clock time, since block order is
                                            # what's authoritative on-chain
        "timestamp_ms": <int, optional>,   # if the indexer also gives you a block
                                            # timestamp, keep it for convenience --
                                            # not used for ordering, only for filtering
                                            # "events up to this wall-clock instant"
    }

If your actual event rows use different field names, the cheapest fix is to
rename them into this shape right where you fetch them, rather than
threading different names through the replay logic below.
"""
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Literal, Optional

Outcome = Literal["YES", "NO"]
Side = Literal["bid", "ask"]


@dataclass
class OrderBookState:
    """Resting orders on one side of one outcome leg, at a point in time."""
    bids: dict  # price -> total resting quantity
    asks: dict  # price -> total resting quantity

    def best_bid(self) -> Optional[float]:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> Optional[float]:
        return min(self.asks) if self.asks else None

    def mid(self) -> Optional[float]:
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        m = (bid + ask) / 2
        return m if 0 < m < 1 else None

    def bound(self) -> Optional[float]:
        """One-sided fallback, same convention as signal.ts's marketBoundUp:
        an ask caps fair value, a bid floors it. Use when mid() is None."""
        ask, bid = self.best_ask(), self.best_bid()
        p = ask if ask is not None else bid
        return p if p is not None and 0 < p < 1 else None


class MarketReplay:
    """
    Replays one market's order events in block order and answers "what did
    the YES/NO book look like at block B (or wall-clock time T)?" without
    re-scanning the full event list on every query -- events are sorted once,
    then a running book is rebuilt incrementally as you query forward in time.

    Query in ASCENDING time order (matches how simulate.py already walks
    windows chronologically) for O(events) total instead of O(events *
    windows). Querying backward or out of order still works, it just
    re-replays from the start, which is correct but slow -- see
    book_at_block() docstring.
    """

    def __init__(self, events: list[dict]):
        # Stable sort by block so same-block events (e.g. a fill immediately
        # after its own placement) keep their original relative order.
        self.events = sorted(events, key=lambda e: e["block"])
        self.blocks = [e["block"] for e in self.events]

        # order_id -> (outcome, side, price, remaining_qty), for O(1) lookup
        # when a Cancelled/Filled/Reduced event needs to find its resting order.
        self._resting: dict[str, dict] = {}
        self._books: dict[Outcome, OrderBookState] = {
            "YES": OrderBookState(bids={}, asks={}),
            "NO": OrderBookState(bids={}, asks={}),
        }
        self._cursor = 0  # index into self.events of the next event to apply

    def _apply(self, e: dict) -> None:
        outcome = e["outcome"]
        book = self._books[outcome]
        side_book = book.bids if e["side"] == "bid" else book.asks
        oid = e["order_id"]
        px = e["price"]

        if e["event_type"] == "OrderPlaced":
            qty = e["quantity"]
            self._resting[oid] = {"outcome": outcome, "side": e["side"], "price": px, "qty": qty}
            side_book[px] = side_book.get(px, 0.0) + qty

        elif e["event_type"] in ("OrderCancelled", "OrderFilled"):
            resting = self._resting.pop(oid, None)
            if resting is None:
                return  # event for an order we never saw placed -- ignore rather than crash
            rb = self._books[resting["outcome"]].bids if resting["side"] == "bid" else self._books[resting["outcome"]].asks
            remaining = rb.get(resting["price"], 0.0) - resting["qty"]
            if remaining <= 1e-9:
                rb.pop(resting["price"], None)
            else:
                rb[resting["price"]] = remaining

        elif e["event_type"] == "OrderReduced":
            resting = self._resting.get(oid)
            if resting is None:
                return
            new_qty = e["quantity"]  # remaining quantity after the reduction
            rb = self._books[resting["outcome"]].bids if resting["side"] == "bid" else self._books[resting["outcome"]].asks
            delta = new_qty - resting["qty"]
            rb[resting["price"]] = rb.get(resting["price"], 0.0) + delta
            if rb[resting["price"]] <= 1e-9:
                rb.pop(resting["price"], None)
            resting["qty"] = new_qty

    def book_at_block(self, block: int) -> dict:
        """
        Returns {"YES": OrderBookState, "NO": OrderBookState} as of the given
        block (inclusive). Call with non-decreasing `block` across a session
        for O(events) total; an out-of-order call rewinds and replays from
        the start, which is correct but O(events) EVERY time -- fine for a
        few markets, not for scanning thousands out of order.
        """
        target_idx = bisect_right(self.blocks, block)
        if target_idx < self._cursor:
            # Rewinding: cheapest correct option is a full replay from zero
            # rather than trying to "undo" events, since cancels/fills don't
            # carry enough info to reverse cleanly.
            self._resting.clear()
            self._books = {"YES": OrderBookState(bids={}, asks={}), "NO": OrderBookState(bids={}, asks={})}
            self._cursor = 0

        while self._cursor < target_idx:
            self._apply(self.events[self._cursor])
            self._cursor += 1

        return self._books


def fetch_order_events(market_id: str) -> list[dict]:
    """
    *** THIS IS THE FUNCTION YOU NEED TO FILL IN. ***

    Should return every OrderPlaced/OrderCancelled/OrderFilled/OrderReduced
    event for `market_id`'s YES and NO books, each reshaped into the dict
    shape documented at the top of this file.

    Two ways to implement it, per what the docs you pasted said:

    1. @somnia-chain/markets-sdk indexer (docs called this "best supported"
       for Event Contracts). This is a JS/TS package per the bot-kit repo --
       if it doesn't expose a plain HTTP/GraphQL endpoint you can hit from
       Python, the simplest bridge is a tiny Node script that dumps one
       market's order history to JSON, and this function just reads that
       JSON off disk. Check docs.dreamdex.io/developers for the indexer's
       actual query shape -- I don't have it and won't guess at field names.

    2. Raw on-chain events via Somnia RPC (OrderPlaced/OrderCancelled/
       OrderFilled/OrderReduced from the SpotPool/EventContract logs) using
       web3.py + the contract ABI, or via an explorer/Dune export. More work,
       but doesn't depend on the indexer's query shape at all.

    Paste me an example response from whichever route you use and I'll write
    the exact field-mapping into this function.
    """
    raise NotImplementedError(
        "Fill in with a real indexer/RPC call — see docstring for the two options."
    )


def market_prob_up_at(replay: MarketReplay, block: int) -> Optional[float]:
    """
    The market's own P(up) from the YES book at a given block -- mid if the
    book is two-sided, the one-sided bound otherwise, None if there's nothing
    resting on the YES book yet. Same convention as ec-oracle-follow's
    marketImpliedUp/marketBoundUp in signal.ts, so results are directly
    comparable to what the live bot would have seen.
    """
    yes_book = replay.book_at_block(block)["YES"]
    mid = yes_book.mid()
    return mid if mid is not None else yes_book.bound()
