"""
일일티커 v2 — Main generation script (Probability Model)

Pipeline:
  1. Load data
  2. Train probability model on all available history
  3. Screen today's candidates through gate
  4. Predict probabilities for each threshold (+1.5%, +2%, +2.5%, +3%)
  5. Sort by +2% probability, output top 10
  6. Write site/data/latest.json

Usage:
    python scripts/generate.py                    # use latest date in DB
    python scripts/generate.py --date 2026-02-12  # specific date
    python scripts/generate.py --update           # fetch new data first
"""

import json
import sys
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scanner.data_fetcher import DailyDataFetcher
from scanner.probability_model import (
    ProbabilityModel, preload_data, _screen_and_extract, FEATURE_NAMES, THRESHOLDS
)

DB_PATH = "data/daily_cache.db"
MODEL_PATH = "data/probability_model.pkl"
OUTPUT_PATH = "site/data/latest.json"

# Loose gate params — let more candidates through, model differentiates
GATE_PARAMS = {
    "volume_min": 1.0,
    "breakout_min": -3.0,
    "breakout_max": 20.0,
    "foreigner_trend_min": 2,
    "require_institution": False,
}

MAX_OUTPUT = 10
SORT_THRESHOLD = 2.0  # Sort by probability of hitting +2%


def main():
    parser = argparse.ArgumentParser(description="일일티커 v2 — Daily Recommendations")
    parser.add_argument("--date", type=str, default=None,
                        help="Target date (YYYY-MM-DD). Default: latest in DB.")
    parser.add_argument("--update", action="store_true",
                        help="Fetch new data from KRX before screening.")
    parser.add_argument("--retrain", action="store_true",
                        help="Force model retrain (otherwise loads cached model).")
    args = parser.parse_args()

    print("=" * 55)
    print("  일일티커 v2 — Probability Recommendations")
    print("=" * 55)

    # Step 1: Data
    fetcher = DailyDataFetcher(DB_PATH)

    if args.update:
        print("\n[1/4] Updating data from KRX...")
        fetcher.update()
    else:
        print("\n[1/4] Using cached data.")

    fetcher.stats()

    # Determine target date
    if args.date:
        target_date = args.date
    else:
        date_range = fetcher.get_date_range()
        if not date_range[1]:
            print("ERROR: No data in database. Run with --update first.")
            return
        target_date = date_range[1]

    print(f"  Target date: {target_date}")

    # Step 2: Load or train model
    print("\n[2/4] Preparing model...")
    data = preload_data(DB_PATH)

    # Validate target_date is an actual trading day with data
    # If not (holiday/weekend with sparse data), walk backwards
    all_dates = data["all_dates"]
    if target_date not in all_dates:
        # Find the most recent trading day at or before target_date
        candidates_dates = [d for d in all_dates if d <= target_date]
        if not candidates_dates:
            print("ERROR: No trading days found at or before target date.")
            return
        target_date = candidates_dates[-1]
        print(f"  Adjusted to last trading day: {target_date}")

    model_file = Path(MODEL_PATH)
    if model_file.exists() and not args.retrain:
        model = ProbabilityModel()
        model.load(MODEL_PATH)
    else:
        print("  Training new model on all available data...")
        model = ProbabilityModel()
        all_train_dates = [d for d in data["all_dates"] if d <= target_date]
        model.train(data, all_train_dates, gate_params=GATE_PARAMS)
        model.save(MODEL_PATH)

    if not model.trained:
        print("ERROR: Model training failed.")
        return

    # Step 3: Screen today + predict
    print(f"\n[3/4] Screening {target_date} and predicting...")
    candidates = _screen_and_extract(target_date, data, GATE_PARAMS)

    if not candidates:
        print("  No candidates found.")
        _write_empty(target_date)
        return

    print(f"  {len(candidates)} candidates passed gate")

    # Predict probabilities for each candidate
    results = []
    for cand in candidates:
        features = np.array(cand["features"])
        probs = model.predict(features)

        result = {
            "ticker": cand["ticker"],
            "features": cand["feature_dict"],
            "probabilities": {t: round(float(p) * 100, 1) for t, p in probs.items()},
        }
        results.append(result)

    # Sort by +2% probability descending
    results.sort(key=lambda r: r["probabilities"][SORT_THRESHOLD], reverse=True)

    # Top N for main display
    top = results[:MAX_OUTPUT]

    print(f"\n  === TOP {len(top)} CANDIDATES ===")
    print(f"  {'Rank':<5} {'Ticker':<8} {'1.5%':>6} {'2.0%':>6} {'2.5%':>6} {'3.0%':>6}  {'chg':>6} {'vol':>5} {'gap20':>6}")
    print(f"  {'─' * 60}")
    for i, r in enumerate(top):
        p = r["probabilities"]
        f = r["features"]
        print(f"  {i+1:<5} {r['ticker']:<8} "
              f"{p[1.5]:>5.1f}% {p[2.0]:>5.1f}% {p[2.5]:>5.1f}% {p[3.0]:>5.1f}%  "
              f"{f['change_pct']:>+5.1f}% {f['volume_ratio']:>4.1f}x {f['gap_from_ma20_pct']:>+5.1f}%")

    # Step 4: Write JSON
    print(f"\n[4/4] Writing output...")

    # Format ALL candidates (for search), flag top N as featured
    all_entries = _format_output(results)
    for i, entry in enumerate(all_entries):
        entry["featured"] = i < MAX_OUTPUT
        entry["rank"] = i + 1

    _fill_ticker_names(all_entries)

    # Count screened
    screened_count = sum(
        1 for td in data["ticker_data"].values()
        if target_date in td["dates"]
    )

    output = {
        "version": 2,
        "date": target_date,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "screened_count": screened_count,
        "candidates_count": len(candidates),
        "sort_by": f"+{SORT_THRESHOLD}% 확률",
        "is_stale": target_date < datetime.now().strftime("%Y-%m-%d"),
        "recommendations": [e for e in all_entries if e["featured"]],
        "all_candidates": all_entries,
    }

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  Written to {out_path}")
    print(f"\n{'=' * 55}")
    print(f"  Done! {len(top)} recommendations")
    print(f"{'=' * 55}")


def _format_output(results: list) -> list:
    """Format results for JSON output."""
    entries = []
    for r in results:
        p = r["probabilities"]
        f = r["features"]

        # Strength grade based on +2% probability
        p2 = p[2.0]
        if p2 >= 50:
            grade = "A"
        elif p2 >= 40:
            grade = "B"
        elif p2 >= 30:
            grade = "C"
        else:
            grade = "D"

        # Risk flags
        risks = []
        if f["gap_from_ma20_pct"] > 15:
            risks.append("MA20 과이격")
        if f["upper_shadow_ratio"] > 0.6:
            risks.append("윗꼬리 주의")
        if f["foreigner_intensity"] > 0.3:
            risks.append("외국인 과매수")
        if f["volume_ratio"] > 4.0:
            risks.append("거래량 과열")
        if f["breakout_distance"] > 10:
            risks.append("추격매수 주의")

        entry = {
            "ticker": r["ticker"],
            "name": "",
            "grade": grade,
            "prob_1_5": p[1.5],
            "prob_2_0": p[2.0],
            "prob_2_5": p[2.5],
            "prob_3_0": p[3.0],
            # Key features for display
            "change_pct": round(f["change_pct"], 2),
            "volume_ratio": round(f["volume_ratio"], 2),
            "gap_from_ma20_pct": round(f["gap_from_ma20_pct"], 2),
            "foreigner_trend": int(f["foreigner_trend"]),
            "institution_buying": bool(f["institution_buying"]),
            "upper_shadow_ratio": round(f["upper_shadow_ratio"], 3),
            "return_5d": round(f["return_5d"], 2),
            "foreigner_intensity": round(f["foreigner_intensity"], 4),
            "risks": risks,
        }
        entries.append(entry)

    return entries


def _write_empty(target_date: str):
    output = {
        "version": 2,
        "date": target_date,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "screened_count": 0,
        "candidates_count": 0,
        "sort_by": f"+{SORT_THRESHOLD}% 확률",
        "recommendations": [],
    }
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  Written empty output to {out_path}")


def _fill_ticker_names(entries: list):
    """Look up ticker names via pykrx."""
    try:
        from pykrx import stock as pykrx_stock
    except ImportError:
        return

    seen = {}
    for entry in entries:
        ticker = entry["ticker"]
        if ticker in seen:
            entry["name"] = seen[ticker]
            continue
        try:
            name = pykrx_stock.get_market_ticker_name(ticker)
            entry["name"] = name or ""
            seen[ticker] = entry["name"]
        except Exception:
            entry["name"] = ""
            seen[ticker] = ""


if __name__ == "__main__":
    main()