"""
Metrics computed from a backtest's trades DataFrame (see backtest/simulate.py).
"""
import numpy as np
import pandas as pd


def compute_metrics(all_windows: pd.DataFrame, trades: pd.DataFrame) -> dict:
    n_windows = len(all_windows)
    n_trades = len(trades)

    if n_trades == 0:
        return {
            "windows_evaluated": n_windows,
            "trades_taken": 0,
            "participation_rate": 0.0,
            "win_rate": None,
            "total_pnl": 0.0,
            "avg_pnl_per_trade": None,
            "max_drawdown": 0.0,
            "sharpe_like": None,
        }

    win_rate = trades["won"].mean()
    total_pnl = trades["pnl"].sum()
    avg_pnl = trades["pnl"].mean()

    cumulative = trades["pnl"].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min()

    pnl_std = trades["pnl"].std()
    sharpe_like = (avg_pnl / pnl_std) * np.sqrt(len(trades)) if pnl_std and pnl_std > 0 else None

    return {
        "windows_evaluated": n_windows,
        "trades_taken": n_trades,
        "participation_rate": n_trades / n_windows if n_windows else 0.0,
        "win_rate": float(win_rate),
        "total_pnl": float(total_pnl),
        "avg_pnl_per_trade": float(avg_pnl),
        "max_drawdown": float(max_drawdown),
        "sharpe_like": float(sharpe_like) if sharpe_like is not None else None,
    }


def breakdown_by_hour(trades: pd.DataFrame) -> pd.DataFrame:
    """Win rate / avg pnl by hour of day — useful chart for the demo."""
    if len(trades) == 0:
        return pd.DataFrame(columns=["hour", "trades", "win_rate", "avg_pnl"])

    df = trades.copy()
    df["hour"] = pd.to_datetime(df["time"]).dt.hour
    grouped = df.groupby("hour").agg(
        trades=("pnl", "count"),
        win_rate=("won", "mean"),
        avg_pnl=("pnl", "mean"),
    ).reset_index()
    return grouped


def breakdown_by_regime(all_windows: pd.DataFrame) -> pd.DataFrame:
    """Shows how many windows fell into each volatility regime — proves the filter does something."""
    return all_windows.groupby("vol_regime").size().reset_index(name="count")
