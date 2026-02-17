"""
일일티커 v2 — History Tracker

Checks if yesterday's recommendations actually hit their predicted thresholds.
Appends results to site/data/history.json for the verification tab.

Run daily AFTER market close (after 15:30 KST):
    python scripts/track_history.py

Flow:
  1. Read yesterday's latest.json (archived)
  2. For each recommended ticker, check: did today's price hit +2% from open?
  3. Append to history.json
"""

import json
import sys
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DB_PATH = "data/daily_cache.db"
LATEST_PATH = "site/data/latest.json"
HISTORY_PATH = "site/data/history.json"
ARCHIVE_DIR = "site/data/archive"
TARGET_THRESHOLD = 2.0  # Track +2% hit rate


def main():
    print("=" * 50)
    print("  일일티커 — History Tracker")
    print("=" * 50)

    # Step 1: Find the most recent archived prediction
    archive_dir = Path(ARCHIVE_DIR)
    if not archive_dir.exists():
        archive_dir.mkdir(parents=True, exist_ok=True)

    # Archive current latest.json before it gets overwritten
    latest_path = Path(LATEST_PATH)
    if latest_path.exists():
        with open(latest_path, "r", encoding="utf-8") as f:
            latest = json.load(f)

        pred_date = latest.get("date", "")
        if pred_date:
            archive_file = archive_dir / f"{pred_date}.json"
            if not archive_file.exists():
                with open(archive_file, "w", encoding="utf-8") as f:
                    json.dump(latest, f, ensure_ascii=False, indent=2)
                print(f"  Archived {pred_date} predictions")

    # Step 2: Find predictions that need verification
    # Look for archived predictions where we have next-day OHLCV data
    conn = sqlite3.connect(DB_PATH)
    all_dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT date FROM daily_ohlcv ORDER BY date"
    ).fetchall()]

    # Load existing history
    history_path = Path(HISTORY_PATH)
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = {"days": []}

    verified_dates = {d["date"] for d in history["days"]}

    # Check each archived prediction
    new_verifications = 0
    archive_files = sorted(archive_dir.glob("*.json"))

    for af in archive_files:
        pred_date = af.stem  # e.g., "2026-02-13"

        if pred_date in verified_dates:
            continue

        # Find next trading day after pred_date
        try:
            pred_idx = all_dates.index(pred_date)
        except ValueError:
            continue

        next_idx = pred_idx + 1
        if next_idx >= len(all_dates):
            continue  # No next-day data yet

        next_date = all_dates[next_idx]

        # Load prediction
        with open(af, "r", encoding="utf-8") as f:
            pred = json.load(f)

        recs = pred.get("recommendations", [])
        if not recs:
            continue

        # Check each ticker
        day_results = []
        for rec in recs:
            ticker = rec["ticker"]
            predicted_prob = rec.get("prob_2_0", 0)

            # Get next day's OHLCV
            row = conn.execute(
                "SELECT open, high, low, close FROM daily_ohlcv WHERE ticker=? AND date=?",
                (ticker, next_date)
            ).fetchone()

            if not row:
                continue

            next_open, next_high, next_low, next_close = row

            if next_open <= 0:
                continue

            # Did it hit +2% from open?
            max_gain_pct = (next_high - next_open) / next_open * 100
            close_gain_pct = (next_close - next_open) / next_open * 100
            hit = max_gain_pct >= TARGET_THRESHOLD

            day_results.append({
                "ticker": ticker,
                "name": rec.get("name", ""),
                "predicted_prob": round(predicted_prob, 1),
                "grade": rec.get("grade", ""),
                "actual_high_pct": round(max_gain_pct, 2),
                "actual_close_pct": round(close_gain_pct, 2),
                "hit": hit,
            })

        if day_results:
            hits = sum(1 for r in day_results if r["hit"])
            total = len(day_results)

            history["days"].append({
                "date": pred_date,
                "next_trading_date": next_date,
                "total": total,
                "hits": hits,
                "hit_rate": round(hits / total * 100, 1) if total > 0 else 0,
                "results": day_results,
            })

            new_verifications += 1
            print(f"  {pred_date}: {hits}/{total} hit +{TARGET_THRESHOLD}% ({hits/total*100:.0f}%)")

    conn.close()

    # Sort by date descending (newest first)
    history["days"].sort(key=lambda d: d["date"], reverse=True)

    # Compute aggregate stats
    total_all = sum(d["total"] for d in history["days"])
    hits_all = sum(d["hits"] for d in history["days"])
    history["aggregate"] = {
        "total_predictions": total_all,
        "total_hits": hits_all,
        "hit_rate": round(hits_all / total_all * 100, 1) if total_all > 0 else 0,
        "days_tracked": len(history["days"]),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    # Save
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"\n  New verifications: {new_verifications}")
    print(f"  Total tracked: {len(history['days'])} days, {total_all} predictions")
    if total_all > 0:
        print(f"  Aggregate hit rate: {hits_all}/{total_all} = {hits_all/total_all*100:.1f}%")
    print(f"  Saved to {history_path}")


if __name__ == "__main__":
    main()