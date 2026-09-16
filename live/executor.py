"""
Thin wrapper around DreamDexEventContractsClient for placing/exiting trades,
kept separate from agent.py so the decision logic and execution logic can be
tested independently (e.g. swap in a mock client in tests).
"""
from live.client import DreamDexEventContractsClient


class Executor:
    def __init__(self, client: DreamDexEventContractsClient):
        self.client = client

    def enter(self, contract_id: str, side: str, amount: float) -> dict:
        return self.client.place_stake(contract_id, side, amount)

    def exit_early(self, contract_id: str, position_id: str) -> dict:
        return self.client.sell_early(contract_id, position_id)

    def read_settlement(self, contract_id: str) -> dict:
        return self.client.get_settlement(contract_id)
