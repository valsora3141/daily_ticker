"""
일일티커 — Screener (Hold-Only)

Filters optimized by grid search (250-day sweep):
  1. MA5/20/60 aligned (close above all three)
  2. Foreigner net buy > 0 today
  3. Foreigner trend ≥ 4/5 days positive
  4. Institution net buy > 0 today
  5. Volume ratio ≥ 1.5x of 20-day avg
  6. Breakout proximity: -2% to +20%
  7. Liquidity floor (avg vol 20d ≥ 50k, price ≥ 1000원)

Grid search result: T5 WR=54%, avg +1.84%, EV=+1.84
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from collections import defaultdict


# ─────────────────────────────────────────────────────────────
# Configuration (grid search optimal)
# ─────────────────────────────────────────────────────────────

MIN_PRICE = 1_000
MIN_AVG_VOLUME = 50_000
VOLUME_RATIO_MIN = 1.5
BREAKOUT_MIN_PCT = -2.0
BREAKOUT_MAX_PCT = 20.0
FOREIGNER_TREND_MIN = 4


def screen(data: Dict) -> List[Dict]:
    """
    Run hold screening pipeline.

    Args:
        data: dict from DailyDataFetcher.get_screening_data()

    Returns:
        List of candidate dicts sorted by raw_score descending.
    """
    if "error" in data:
        print(f"  Screener error: {data['error']}")
        return []

    target_date = data["target_date"]
    ohlcv = data["ohlcv"]
    investor_today = data["investor_today"]
    investor_recent = data["investor_recent"]

    contexts = _build_contexts(ohlcv, target_date)

    inv_today_map = {}
    if not investor_today.empty:
        for _, row in investor_today.iterrows():
            inv_today_map[row["ticker"]] = {
                "foreigner_net": row["foreigner_net"],
                "institution_net": row["institution_net"],
                "individual_net": row["individual_net"],
            }

    foreigner_trend_map = _build_foreigner_trend(investor_recent)

    candidates = []
    rejections = defaultdict(int)

    for ticker, ctx in contexts.items():
        inv = inv_today_map.get(ticker, {})
        ftrend = foreigner_trend_map.get(ticker, {
            "positive_days": 0, "total_days": 0, "cumulative_net": 0
        })

        result = _evaluate(ticker, ctx, inv, ftrend)

        if result["pass_all"]:
            candidates.append(result)
        else:
            rejections[result["rejection"]] += 1

    candidates.sort(key=lambda c: c["raw_score"], reverse=True)

    print(f"  Screened {len(contexts)} tickers for {target_date}")
    print(f"  Candidates: {len(candidates)}")
    if rejections:
        print(f"  Rejections: {dict(rejections)}")

    return candidates


def _build_contexts(ohlcv: pd.DataFrame, target_date: str) -> Dict:
    """Compute MAs, yesterday's high/low, volume stats for all tickers."""
    contexts = {}
    grouped = ohlcv.groupby("ticker")

    for ticker, group in grouped:
        group = group.sort_values("date")

        if len(group) < 5:
            continue

        today_row = group[group["date"] == target_date]
        if today_row.empty:
            continue

        today = today_row.iloc[0]

        if today["close"] < MIN_PRICE:
            continue

        history = group[group["date"] < target_date].sort_values("date", ascending=False)
        if len(history) < 5:
            continue

        yesterday = history.iloc[0]
        closes = history["close"].values
        volumes = history["volume"].values

        ma5 = np.mean(closes[:5]) if len(closes) >= 5 else 0
        ma20 = np.mean(closes[:20]) if len(closes) >= 20 else 0
        ma60 = np.mean(closes[:60]) if len(closes) >= 60 else 0

        avg_volume_20 = np.mean(volumes[:20]) if len(volumes) >= 20 else np.mean(volumes)

        breakout_distance = (today["close"] - yesterday["high"]) / yesterday["high"] * 100

        contexts[ticker] = {
            "ticker": ticker,
            "date": target_date,
            "close": today["close"],
            "open": today["open"],
            "high": today["high"],
            "low": today["low"],
            "volume": today["volume"],
            "change_pct": today.get("change_pct", 0),
            "yesterday_high": yesterday["high"],
            "yesterday_low": yesterday["low"],
            "yesterday_close": yesterday["close"],
            "ma5": ma5,
            "ma20": ma20,
            "ma60": ma60,
            "avg_volume_20": avg_volume_20,
            "volume_ratio": today["volume"] / avg_volume_20 if avg_volume_20 > 0 else 0,
            "breakout_distance": breakout_distance,
            "close_above_ma5": today["close"] > ma5,
            "close_above_ma20": today["close"] > ma20,
            "close_above_ma60": today["close"] > ma60,
        }

    return contexts


def _build_foreigner_trend(investor_recent: pd.DataFrame) -> Dict:
    trend = {}
    if investor_recent.empty:
        return trend

    grouped = investor_recent.groupby("ticker")
    for ticker, group in grouped:
        positive_days = (group["foreigner_net"] > 0).sum()
        total_days = len(group)
        trend[ticker] = {
            "positive_days": int(positive_days),
            "total_days": int(total_days),
            "cumulative_net": float(group["foreigner_net"].sum()),
        }

    return trend


def _evaluate(ticker: str, ctx: Dict, inv: Dict, ftrend: Dict) -> Dict:
    """Evaluate hold filters for a single ticker."""
    foreigner_net = inv.get("foreigner_net", 0)
    institution_net = inv.get("institution_net", 0)
    foreigner_trend_days = ftrend.get("positive_days", 0)
    foreigner_cumulative = ftrend.get("cumulative_net", 0)

    pass_ma = ctx["close_above_ma5"] and ctx["close_above_ma20"] and ctx["close_above_ma60"]
    foreigner_buying = foreigner_net > 0
    institution_buying = institution_net > 0
    volume_ratio = ctx["volume_ratio"]
    liquidity_ok = ctx["avg_volume_20"] >= MIN_AVG_VOLUME
    breakout_distance = ctx["breakout_distance"]

    # Filter chain
    rejection = None
    if not pass_ma:
        rejection = "MA"
    elif not foreigner_buying:
        rejection = "FOREIGNER_TODAY"
    elif foreigner_trend_days < FOREIGNER_TREND_MIN:
        rejection = "FOREIGNER_TREND"
    elif not institution_buying:
        rejection = "INSTITUTION"
    elif volume_ratio < VOLUME_RATIO_MIN:
        rejection = "VOLUME"
    elif breakout_distance < BREAKOUT_MIN_PCT:
        rejection = "BREAKOUT_FAR"
    elif breakout_distance > BREAKOUT_MAX_PCT:
        rejection = "BREAKOUT_EXTENDED"
    elif not liquidity_ok:
        rejection = "LIQUIDITY"

    pass_all = rejection is None

    # Raw score for sorting
    score = 0.0
    if pass_all:
        # Foreigner trend is primary
        score += min(foreigner_trend_days * 0.8, 4.0)
        # Institution alignment
        score += 1.5
        # Volume
        score += min(volume_ratio * 0.5, 2.0)
        # Breakout proximity
        if 0 <= breakout_distance <= 5:
            score += 1.5
        elif -2 <= breakout_distance < 0:
            score += 1.0
        elif 5 < breakout_distance <= 15:
            score += 0.5
        # Cumulative foreigner buying
        if foreigner_cumulative > 0:
            score += 0.5

    return {
        "ticker": ticker,
        "date": ctx["date"],
        "close": ctx["close"],
        "open": ctx["open"],
        "high": ctx["high"],
        "low": ctx["low"],
        "volume": ctx["volume"],
        "change_pct": ctx["change_pct"],
        "yesterday_high": ctx["yesterday_high"],
        "yesterday_close": ctx["yesterday_close"],
        "ma5": round(ctx["ma5"], 2),
        "ma20": round(ctx["ma20"], 2),
        "ma60": round(ctx["ma60"], 2),
        "avg_volume_20": ctx["avg_volume_20"],
        "volume_ratio": round(volume_ratio, 2),
        "breakout_distance": round(breakout_distance, 2),
        "foreigner_net": foreigner_net,
        "institution_net": institution_net,
        "foreigner_trend_days": foreigner_trend_days,
        "foreigner_cumulative": foreigner_cumulative,
        "institution_buying": institution_buying,
        "pass_all": pass_all,
        "rejection": rejection,
        "raw_score": round(score, 2),
    }