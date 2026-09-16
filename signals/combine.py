"""
Combines imbalance + momentum into a model probability, compares against the
market's implied probability (the Event Contract's own ask price, or a
synthetic proxy in backtest), and produces a trade decision.
"""
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Decision:
    side: Optional[str]        # "UP", "DOWN", or None (sit out)
    model_prob_up: float
    market_prob_up: float
    edge: float
    vol_regime: str
    imbalance: float
    momentum_signal: float
    reason: str

    def to_dict(self):
        return asdict(self)


def combine_signals(imbalance: float, momentum_signal: float, w1: float = 0.5, w2: float = 0.5,
                     clamp_range: float = 0.45) -> float:
    """Returns model_prob_up in [0.5 - clamp_range, 0.5 + clamp_range]."""
    edge_estimate = w1 * imbalance + w2 * momentum_signal
    edge_estimate = max(-clamp_range, min(clamp_range, edge_estimate))
    return 0.5 + edge_estimate


def decide(imbalance: float, momentum_signal: float, vol_regime: str, market_prob_up: float,
           w1: float = 0.5, w2: float = 0.5, entry_threshold: float = 0.06) -> Decision:
    model_prob_up = combine_signals(imbalance, momentum_signal, w1, w2)
    edge = model_prob_up - market_prob_up

    if vol_regime == "SIT_OUT":
        return Decision(None, model_prob_up, market_prob_up, edge, vol_regime,
                         imbalance, momentum_signal, "sit out: volatility regime")

    if edge > entry_threshold:
        side = "UP"
        reason = f"edge {edge:+.3f} > threshold {entry_threshold:.3f}"
    elif edge < -entry_threshold:
        side = "DOWN"
        reason = f"edge {edge:+.3f} < -threshold {entry_threshold:.3f}"
    else:
        side = None
        reason = f"edge {edge:+.3f} inside threshold band, no trade"

    return Decision(side, model_prob_up, market_prob_up, edge, vol_regime,
                     imbalance, momentum_signal, reason)


def check_exit(current_edge: float, entry_side: str, exit_threshold: float = 0.05) -> bool:
    """
    Early-exit trigger: exit if the live edge has flipped sign relative to the
    entry side and the magnitude clears exit_threshold.
    """
    if entry_side == "UP" and current_edge < -exit_threshold:
        return True
    if entry_side == "DOWN" and current_edge > exit_threshold:
        return True
    return False
