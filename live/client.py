"""
DreamDEX Event Contracts client — STUB.

BLOCKED on confirming (see project doc section 0 "not yet confirmed"):
  - Exact REST/WebSocket endpoints for Event Contracts (NOT the spot CLOB
    endpoints from the bot-kit repo — likely a separate namespace, e.g.
    /v0/event-contracts/... — confirm in docs.dreamdex.io/developers or Telegram)
  - Contract address(es) + ABI for staking / early sell-back
  - "Published settlement reference": which price feed/oracle, update frequency
  - Rate limits, session-key setup specific to Event Contracts
  - Whether testnet (Shannon, chain 50312) has Event Contracts live

Fill in each method below once those are confirmed. Until then, every method
raises NotImplementedError so nothing silently no-ops in a live run.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class EventContractWindow:
    contract_id: str
    asset: str              # "BTC" | "ETH"
    open_price: float
    open_time: str
    close_time: str
    up_ask: float
    up_bid: float
    down_ask: float
    down_bid: float


class DreamDexEventContractsClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None, session_key: Optional[str] = None,
                 dry_run: bool = True):
        self.base_url = base_url
        self.api_key = api_key
        self.session_key = session_key
        self.dry_run = dry_run  # keep True until endpoints + contract details are confirmed

    def get_current_window(self, asset: str, duration: str = "15m") -> EventContractWindow:
        """TODO: GET current open window + live book/ask/bid for Up and Down."""
        raise NotImplementedError("Confirm Event Contracts REST endpoint before wiring this up")

    def get_order_book(self, contract_id: str, side: str, top_n: int = 5) -> dict:
        """TODO: GET order book depth for the Up or Down contract. Returns {'bids': [...], 'asks': [...]}"""
        raise NotImplementedError("Confirm Event Contracts book endpoint before wiring this up")

    def place_stake(self, contract_id: str, side: str, amount: float) -> dict:
        """
        TODO: submit a stake (buy Up or Down) — likely an on-chain tx against
        the Event Contracts contract address, or a signed order to the CLOB.
        Respect self.dry_run: if True, log the intended action instead of sending.
        """
        if self.dry_run:
            print(f"[DRY_RUN] would stake {amount} on {side} for {contract_id}")
            return {"dry_run": True, "contract_id": contract_id, "side": side, "amount": amount}
        raise NotImplementedError("Confirm Event Contracts staking endpoint/contract before going live")

    def sell_early(self, contract_id: str, position_id: str) -> dict:
        """TODO: submit an early-exit sell against a live position."""
        if self.dry_run:
            print(f"[DRY_RUN] would sell early: {position_id} ({contract_id})")
            return {"dry_run": True, "position_id": position_id}
        raise NotImplementedError("Confirm Event Contracts early-exit endpoint before going live")

    def get_settlement(self, contract_id: str) -> dict:
        """TODO: read final settlement result once window closes."""
        raise NotImplementedError("Confirm settlement reference/oracle endpoint before wiring this up")
