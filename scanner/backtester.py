"""
일일티커 — Backtester (Hold-Only)

For each candidate, finds historical dates with similar setups,
checks what happened at T+1, T+3, T+5, T+10 from next-day open entry.

Entry assumption: buy at next day's open (recommendation seen after close).
"""

import numpy as np
import pandas as pd
from typing import Dict, List


HOLD_PERIODS = [1, 3, 5, 10]

# Match criteria for historical setups
MATCH_VOLUME_RATIO_MIN = 1.0
MATCH_BREAKOUT_PROXIMITY_PCT = 3.0
MATCH_FOREIGNER_TREND_MIN = 3

MIN_MATCHES_USABLE = 3
MIN_MATCHES_CONFIDENT = 10


def backtest_candidates(
    candidates: List[Dict],
    fetcher,
    lookback_days: int = 1200,
) -> List[Dict]:
    """Backtest all candidates against historical patterns."""
    if not candidates:
        return []

    target_date = candidates[0]["date"]
    print(f"  Backtesting {len(candidates)} candidates against {lookback_days}-day history...")

    results = []

    for i, cand in enumerate(candidates):
        ticker = cand["ticker"]
        bt = _backtest_single(ticker, target_date, fetcher, lookback_days)
        result = {**cand, **bt}
        results.append(result)

        if (i + 1) % 5 == 0 or i == 0:
            t5_wr = bt.get("hold_t5_wr", 0)
            matches = bt["match_count"]
            print(f"    [{i+1}/{len(candidates)}] {ticker}: "
                  f"{matches} matches, T5 WR={t5_wr:.0f}%")

    print(f"  Backtesting complete.")
    return results


def _backtest_single(
    ticker: str,
    target_date: str,
    fetcher,
    lookback_days: int,
) -> Dict:
    """Backtest a single ticker."""
    history = fetcher.get_ticker_history(ticker, n_days=lookback_days, before_date=target_date)

    if len(history) < 65:
        return _empty_result()

    investor = fetcher.get_investor_history(ticker, n_days=lookback_days, before_date=target_date)

    inv_map = {}
    if not investor.empty:
        for _, row in investor.iterrows():
            inv_map[row["date"]] = {
                "foreigner_net": row["foreigner_net"],
                "institution_net": row["institution_net"],
            }

    matches = _find_matches(history, inv_map)

    if len(matches) < MIN_MATCHES_USABLE:
        return _empty_result(match_count=len(matches))

    hold_results = {p: [] for p in HOLD_PERIODS}

    dates = history["date"].tolist()
    date_to_idx = {d: i for i, d in enumerate(dates)}

    for match_date, match_data in matches:
        idx = date_to_idx.get(match_date)
        if idx is None:
            continue

        # Entry = next day's open
        next_idx = idx + 1
        if next_idx >= len(history):
            continue

        next_day = history.iloc[next_idx]
        entry_price = next_day["open"]
        if entry_price <= 0:
            continue

        for period in HOLD_PERIODS:
            future_idx = next_idx + period
            if future_idx < len(history):
                future_close = history.iloc[future_idx]["close"]
                hold_return = (future_close - entry_price) / entry_price * 100
                hold_results[period].append(hold_return)

    result = {
        "match_count": len(matches),
        "confidence": _confidence_level(len(matches)),
    }

    for period in HOLD_PERIODS:
        key = f"hold_t{period}"
        returns = hold_results[period]
        if returns:
            wins = sum(1 for r in returns if r > 0)
            result[f"{key}_wr"] = wins / len(returns) * 100
            result[f"{key}_avg"] = np.mean(returns)
            result[f"{key}_median"] = np.median(returns)
            result[f"{key}_samples"] = len(returns)
        else:
            result[f"{key}_wr"] = 0
            result[f"{key}_avg"] = 0
            result[f"{key}_median"] = 0
            result[f"{key}_samples"] = 0

    return result


def _find_matches(history: pd.DataFrame, inv_map: Dict) -> List:
    """Find historical dates matching hold setup criteria."""
    if len(history) < 65:
        return []

    closes = history["close"].values
    volumes = history["volume"].values
    dates = history["date"].tolist()

    matches = []

    for i in range(60, len(history)):
        row = history.iloc[i]
        d = dates[i]

        if row["close"] <= 0 or row["open"] <= 0:
            continue

        # MA alignment
        ma5 = np.mean(closes[i-5:i])
        ma20 = np.mean(closes[i-20:i])
        ma60 = np.mean(closes[i-60:i])

        if not (row["close"] > ma5 and row["close"] > ma20 and row["close"] > ma60):
            continue

        # Investor data
        inv = inv_map.get(d, {})
        fgn = inv.get("foreigner_net", 0)
        inst = inv.get("institution_net", 0)

        if fgn <= 0:
            continue
        if inst <= 0:
            continue

        # Volume ratio
        avg_vol_20 = np.mean(volumes[i-20:i])
        if avg_vol_20 > 0:
            vol_ratio = row["volume"] / avg_vol_20
        else:
            vol_ratio = 0

        if vol_ratio < MATCH_VOLUME_RATIO_MIN:
            continue

        # Breakout proximity
        if i > 0:
            yesterday_high = history.iloc[i-1]["high"]
            if yesterday_high > 0:
                breakout_dist = (row["close"] - yesterday_high) / yesterday_high * 100
                if breakout_dist < -MATCH_BREAKOUT_PROXIMITY_PCT:
                    continue

        matches.append((d, row))

    return matches


def _confidence_level(match_count: int) -> str:
    if match_count >= 30:
        return "HIGH"
    elif match_count >= MIN_MATCHES_CONFIDENT:
        return "MEDIUM"
    elif match_count >= MIN_MATCHES_USABLE:
        return "LOW"
    else:
        return "INSUFFICIENT"


def _empty_result(match_count: int = 0) -> Dict:
    result = {
        "match_count": match_count,
        "confidence": _confidence_level(match_count),
    }
    for period in HOLD_PERIODS:
        key = f"hold_t{period}"
        result[f"{key}_wr"] = 0
        result[f"{key}_avg"] = 0
        result[f"{key}_median"] = 0
        result[f"{key}_samples"] = 0
    return result