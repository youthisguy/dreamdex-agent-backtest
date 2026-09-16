"""
Main live decision loop (Phase 3). Runs in DRY_RUN by default (logs decisions,
sends no real tx) exactly like the bot-kit pattern, until Event Contract
endpoints are confirmed and live/client.py is filled in.

The decision + logging logic here is real and usable now against a mocked
client; only DreamDexEventContractsClient's internals are blocked.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from live.client import DreamDexEventContractsClient
from signals.momentum import compute_raw_momentum
from signals.combine import decide, check_exit

LOG_PATH = Path(__file__).parent.parent / "logs" / "decisions.jsonl"


def log_decision(record: dict):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record["logged_at"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


class Agent:
    def __init__(self, client: DreamDexEventContractsClient, asset: str = "BTC",
                 stake: float = 10.0, entry_threshold: float = 0.06, poll_seconds: int = 5):
        self.client = client
        self.asset = asset
        self.stake = stake
        self.entry_threshold = entry_threshold
        self.poll_seconds = poll_seconds
        self.open_position = None  # {"side", "contract_id", "entry_edge", ...}

    def evaluate_and_act(self):
        """One decision cycle. TODO: replace placeholder book/imbalance calls
        once live/client.py's get_current_window/get_order_book are implemented."""
        window = self.client.get_current_window(self.asset)

        book = self.client.get_order_book(window.contract_id, side="up")
        bid_vol = sum(size for _, size in book.get("bids", []))
        ask_vol = sum(size for _, size in book.get("asks", []))
        from signals.imbalance import compute_imbalance
        imbalance = compute_imbalance(bid_vol, ask_vol)

        # TODO: real momentum needs a rolling tick buffer; wire to a live
        # price feed and reuse signals/momentum.py's compute_momentum(ticks, ...).
        momentum_signal = compute_raw_momentum(window.up_ask, window.open_price)

        vol_regime = "TRADE"  # TODO: wire signals/volatility.py with live tick buffer
        market_prob_up = window.up_ask

        decision = decide(imbalance, momentum_signal, vol_regime, market_prob_up,
                           entry_threshold=self.entry_threshold)

        log_decision({
            "contract_id": window.contract_id,
            "asset": self.asset,
            **decision.to_dict(),
        })

        if self.open_position is None and decision.side is not None:
            result = self.client.place_stake(window.contract_id, decision.side, self.stake)
            self.open_position = {"side": decision.side, "contract_id": window.contract_id,
                                   "entry_edge": decision.edge, "result": result}
        elif self.open_position is not None:
            if check_exit(decision.edge, self.open_position["side"]):
                result = self.client.sell_early(self.open_position["contract_id"], "position")
                log_decision({"action": "early_exit", **self.open_position, "result": result})
                self.open_position = None

    def run_forever(self):
        print(f"Agent running for {self.asset}, dry_run={self.client.dry_run}")
        while True:
            try:
                self.evaluate_and_act()
            except NotImplementedError as e:
                print(f"Blocked: {e}. Fill in live/client.py once endpoints are confirmed.")
                break
            except Exception as e:
                print(f"Error in decision loop: {e}")
            time.sleep(self.poll_seconds)


if __name__ == "__main__":
    client = DreamDexEventContractsClient(base_url="https://TODO-confirm-base-url", dry_run=True)
    agent = Agent(client)
    agent.run_forever()
