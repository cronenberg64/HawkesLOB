"""LOBSTER limit order book data loader.

Loads LOBSTER message CSV files and extracts timestamps for four event
categories used in the multivariate Hawkes process model.

LOBSTER direction convention (from the LOBSTER readme):
    Direction  1 = Buy limit order   (resting on bid side)
    Direction -1 = Sell limit order   (resting on ask side)

For executions (type 4), the direction refers to the *resting* limit order
that was executed, NOT the aggressor:
    - Execution of a sell limit order (direction -1) means a *buyer* aggressed
      against the ask → this is a Market Buy (MB).
    - Execution of a buy limit order (direction  1) means a *seller* aggressed
      against the bid → this is a Market Sell (MS).

For limit order submissions (type 1), direction indicates which side the
new order sits on:
    - direction  1 → new buy limit order  → LA_B (limit add buy)
    - direction -1 → new sell limit order  → LA_S (limit add sell)

Timestamps are in seconds after midnight; we shift them relative to
market open (34200 = 09:30:00 ET) so t=0 corresponds to the open.
"""

import numpy as np
import pandas as pd

MARKET_OPEN = 34200.0   # 09:30:00 in seconds after midnight
MARKET_CLOSE = 57600.0  # 16:00:00 in seconds after midnight

COLUMNS = ["time", "type", "order_id", "size", "price", "direction"]

EVENT_LABELS = ["MB", "MS", "LA_B", "LA_S"]

# (message_type, direction) → event category
_TYPE_MAP = {
    (4, -1): "MB",    # execution of sell limit = market buy
    (4,  1): "MS",    # execution of buy limit  = market sell
    (1,  1): "LA_B",  # new buy limit order
    (1, -1): "LA_S",  # new sell limit order
}


def load_lobster(message_csv_path: str, levels: int = 10) -> dict[str, np.ndarray]:
    """Load a LOBSTER message CSV and return per-category timestamp arrays.

    Parameters
    ----------
    message_csv_path : str
        Path to the LOBSTER message file, e.g.
        ``AAPL_2012-06-21_34200000_57600000_message_10.csv``.
    levels : int
        Number of order book levels (unused here, included for API symmetry
        with the orderbook loader if one is added later).

    Returns
    -------
    dict[str, np.ndarray]
        Keys are ``"MB"``, ``"MS"``, ``"LA_B"``, ``"LA_S"``. Each value is a
        1-D float64 array of timestamps in seconds relative to market open,
        sorted in ascending order.
    """
    df = pd.read_csv(message_csv_path, header=None, names=COLUMNS)

    # Keep only limit additions (1) and visible executions (4)
    df = df[df["type"].isin([1, 4])].copy()

    # Filter to regular trading hours
    df = df[(df["time"] >= MARKET_OPEN) & (df["time"] <= MARKET_CLOSE)]

    # Map to event category
    df["category"] = df.apply(
        lambda r: _TYPE_MAP.get((r["type"], r["direction"])), axis=1
    )
    df = df.dropna(subset=["category"])

    # Shift to relative time
    df["time"] = df["time"] - MARKET_OPEN

    result = {}
    for label in EVENT_LABELS:
        times = df.loc[df["category"] == label, "time"].values.astype(np.float64)
        times.sort()
        result[label] = times

    total = sum(len(v) for v in result.values())
    print(f"Loaded {total} events from {message_csv_path}")
    for label in EVENT_LABELS:
        print(f"  {label:5s}: {len(result[label]):>7,} events")

    return result
