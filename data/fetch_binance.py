"""
Drop-in replacement for data/fetch_binance.py's fetch_klines, adding:
  - progress printing per request, so you can see it's actually moving
  - a shorter per-request timeout with retry/backoff, so a single flaky
    request doesn't hang indefinitely or kill the whole 180-day fetch
"""
import time
import requests
import pandas as pd

BASE_URL = "https://api.binance.com/api/v3/klines"


def fetch_klines(symbol="BTCUSDT", interval="1m", start_ms=None, end_ms=None,
                  limit=1000, max_retries=3, request_timeout=10):
    rows = []
    request_num = 0

    while True:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_ms:
            params["startTime"] = start_ms

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(BASE_URL, params=params, timeout=request_timeout)
                resp.raise_for_status()
                r = resp.json()
                break
            except (requests.exceptions.RequestException,) as e:
                print(f"  request {request_num} attempt {attempt}/{max_retries} failed: {e}")
                if attempt == max_retries:
                    raise
                time.sleep(2 * attempt)  # backoff: 2s, 4s, ...

        if not r:
            break

        rows.extend(r)
        request_num += 1
        latest = pd.to_datetime(r[-1][0], unit="ms")
        print(f"  fetched {len(rows)} rows so far (up to {latest})", end="\r")

        start_ms = r[-1][0] + 1
        if end_ms and start_ms >= end_ms:
            break
        if len(r) < limit:
            break
        time.sleep(0.3)

    print()  # newline after the \r progress line
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "taker_base", "taker_quote", "ignore"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    return df