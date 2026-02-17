"""
일일티커 — Scorer (Hold-Only)

Score /10 computed from:
  - T+3/T+5 backtest win rate (primary)
  - T+3/T+5 avg return
  - Historical match confidence
  - Foreigner + institution trend strength
  - T+10 persistence (trend durability)
"""

import numpy as np
from typing import Dict, List


def score_and_rank(results: List[Dict], top_n: int = 5) -> List[Dict]:
    """Score and rank candidates, return top N."""
    if not results:
        return []

    scored = []
    for r in results:
        s = _score(r)
        entry = {
            **r,
            "score": round(s, 1),
            "explanation": _explain(r),
        }
        scored.append(entry)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]


def _score(r: Dict) -> float:
    """
    Score a candidate for hold strategy (0-10).

    Components:
      - T+3/T+5 WR (0-3)
      - T+3/T+5 avg return (0-2)
      - Confidence (0-1)
      - Trend strength (0-2)
      - T+10 persistence (0-2)
    """
    score = 0.0

    # 1. Hold WR — average of T+3 and T+5 (0-3)
    t3_wr = r.get("hold_t3_wr", 0)
    t5_wr = r.get("hold_t5_wr", 0)
    t3_samples = r.get("hold_t3_samples", 0)
    t5_samples = r.get("hold_t5_samples", 0)

    if t3_samples >= 3 and t5_samples >= 3:
        avg_wr = (t3_wr + t5_wr) / 2
        score += min(avg_wr / 100 * 3, 3.0)
    elif t3_samples >= 3:
        score += min(t3_wr / 100 * 3, 3.0)
    else:
        score += 1.0

    # 2. Avg hold return (0-2)
    t3_avg = r.get("hold_t3_avg", 0)
    t5_avg = r.get("hold_t5_avg", 0)
    if t3_samples >= 3 and t5_samples >= 3:
        avg_ret = (t3_avg + t5_avg) / 2
        score += min(max(avg_ret / 5.0 * 2, 0), 2.0)
    else:
        score += 0.5

    # 3. Confidence (0-1)
    if r["confidence"] == "HIGH":
        score += 1.0
    elif r["confidence"] == "MEDIUM":
        score += 0.7
    elif r["confidence"] == "LOW":
        score += 0.3

    # 4. Trend strength (0-2)
    trend = 0.0
    if r.get("foreigner_trend_days", 0) >= 5:
        trend += 1.0
    elif r.get("foreigner_trend_days", 0) >= 4:
        trend += 0.7
    if r.get("institution_buying", False):
        trend += 0.5
    if r.get("volume_ratio", 0) >= 2.0:
        trend += 0.3
    if r.get("foreigner_cumulative", 0) > 0:
        trend += 0.2
    score += min(trend, 2.0)

    # 5. T+10 persistence (0-2)
    t10_wr = r.get("hold_t10_wr", 0)
    t10_samples = r.get("hold_t10_samples", 0)
    if t10_samples >= 3:
        if t10_wr >= 60:
            score += 2.0
        elif t10_wr >= 50:
            score += 1.0
        else:
            score += 0.3
    else:
        score += 0.5

    return min(score, 10.0)


def _explain(r: Dict) -> str:
    """One-line explanation of score components."""
    parts = []
    t3 = r.get("hold_t3_samples", 0)
    t5 = r.get("hold_t5_samples", 0)
    if t3 >= 3:
        parts.append(f"T3={r['hold_t3_wr']:.0f}%/{r['hold_t3_avg']:+.1f}%({t3})")
    if t5 >= 3:
        parts.append(f"T5={r['hold_t5_wr']:.0f}%/{r['hold_t5_avg']:+.1f}%({t5})")
    if not parts:
        parts.append("백테스트 데이터 부족")
    parts.append(f"fgn={r.get('foreigner_trend_days', 0)}d")
    if r.get("institution_buying"):
        parts.append("inst=Y")
    return " | ".join(parts)


def format_for_json(ranked: List[Dict]) -> List[Dict]:
    """Format ranked list for latest.json output."""
    output = []

    for r in ranked:
        t3_wr = r.get("hold_t3_wr", 0)
        t5_wr = r.get("hold_t5_wr", 0)
        t3_avg = r.get("hold_t3_avg", 0)
        t5_avg = r.get("hold_t5_avg", 0)

        entry = {
            "ticker": r["ticker"],
            "name": "",
            "price": r["close"],
            "change_pct": round(r.get("change_pct", 0), 2),
            "score": r["score"],
            "setup_type": _classify_setup(r),
            "backtest_wr": round((t3_wr + t5_wr) / 2, 1),
            "backtest_avg_return": round((t3_avg + t5_avg) / 2, 2),
            "backtest_samples": r.get("hold_t5_samples", 0),
            "confidence": r["confidence"],
            "hold_t1_wr": round(r.get("hold_t1_wr", 0), 1),
            "hold_t3_wr": round(t3_wr, 1),
            "hold_t5_wr": round(t5_wr, 1),
            "hold_t10_wr": round(r.get("hold_t10_wr", 0), 1),
            "hold_t3_avg": round(t3_avg, 2),
            "hold_t5_avg": round(t5_avg, 2),
            "volume_ratio": r.get("volume_ratio", 0),
            "foreigner_net": r.get("foreigner_net", 0),
            "foreigner_trend_days": r.get("foreigner_trend_days", 0),
            "institution_buying": r.get("institution_buying", False),
            "breakout_distance": r.get("breakout_distance", 0),
            "risk": _assess_risk(r),
            "explanation": r.get("explanation", ""),
        }
        output.append(entry)

    return output


def _classify_setup(r: Dict) -> str:
    bd = r.get("breakout_distance", 0)
    vol = r.get("volume_ratio", 0)

    if bd >= 0 and vol >= 2.0:
        return "돌파 + 거래량 급증"
    elif bd >= 0:
        return "전고점 돌파"
    elif bd >= -1:
        return "돌파 임박"
    elif vol >= 2.0:
        return "거래량 급증"
    else:
        return "상승추세 지속"


def _assess_risk(r: Dict) -> str:
    risks = []

    if r["confidence"] == "LOW" or r["confidence"] == "INSUFFICIENT":
        risks.append("백테스트 샘플 부족")

    if r.get("breakout_distance", 0) > 10:
        risks.append("이미 급등 (추격매수 주의)")

    t10_wr = r.get("hold_t10_wr", 0)
    if t10_wr < 40 and r.get("hold_t10_samples", 0) >= 3:
        risks.append("T+10 하락 확률 높음")

    if not risks:
        return "특이사항 없음"

    return " / ".join(risks)