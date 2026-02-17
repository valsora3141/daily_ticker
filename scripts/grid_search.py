"""
일일티커 — Parameter Grid Search

Exhaustive sweep of screener parameters against historical data.
For each parameter combo × each trading day:
  1. Run screener with those params
  2. Check actual next-day returns (open-to-high for DL, open-to-close+N for hold)
  3. Aggregate WR, avg return, R:R, candidate count

Usage:
    python scripts/grid_search.py                        # last 250 trading days
    python scripts/grid_search.py --days 500             # last 500 trading days
    python scripts/grid_search.py --strategy day_leech   # only DL sweep
    python scripts/grid_search.py --strategy hold        # only Hold sweep
"""

import sys
import sqlite3
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scanner.data_fetcher import DailyDataFetcher

DB_PATH = "data/daily_cache.db"


# ─────────────────────────────────────────────────────────────
# Parameter Grids
# ─────────────────────────────────────────────────────────────

DL_GRID = {
    "volume_min": [1.0, 1.3, 1.5, 2.0, 2.5],
    "breakout_min": [-2.0, -1.0, 0.0],
    "breakout_max": [2.0, 3.0, 5.0, 8.0],
    "require_foreigner_today": [True],  # always required
}

HOLD_GRID = {
    "volume_min": [0.8, 1.0, 1.3, 1.5],
    "breakout_min": [-3.0, -2.0, -1.0],
    "breakout_max": [8.0, 12.0, 15.0, 20.0],
    "foreigner_trend_min": [2, 3, 4],
    "require_institution": [True, False],
}

# Shared constants
MIN_PRICE = 1_000
MIN_AVG_VOLUME = 50_000
DAY_LEECH_TARGET_PCT = 3.0  # DL "win" if next-day open-to-high >= 3%
HOLD_PERIODS = [1, 3, 5, 10]


# ─────────────────────────────────────────────────────────────
# Data Preloader
# ─────────────────────────────────────────────────────────────

def preload_all_data(db_path: str, n_days: int) -> dict:
    """
    Preload all data into memory for fast iteration.
    Returns dict with everything indexed for quick lookup.
    """
    print(f"  Preloading data...")
    conn = sqlite3.connect(db_path)

    # Get all trading dates
    all_dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT date FROM daily_ohlcv ORDER BY date"
    ).fetchall()]

    # We need 60 days before the test window for MAs
    if len(all_dates) < n_days + 61:
        print(f"  WARNING: Only {len(all_dates)} dates available, need {n_days + 61}")
        n_days = len(all_dates) - 61

    test_dates = all_dates[-(n_days + 1):-1]  # -1 because we need next day for returns
    ma_start_date = all_dates[-(n_days + 61)]

    print(f"  Test window: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")
    print(f"  MA lookback from: {ma_start_date}")

    # Load all OHLCV
    ohlcv = pd.read_sql_query(
        "SELECT * FROM daily_ohlcv WHERE date >= ? ORDER BY ticker, date",
        conn, params=[ma_start_date]
    )

    # Load all investor data
    investor = pd.read_sql_query(
        "SELECT * FROM daily_investor WHERE date >= ? ORDER BY ticker, date",
        conn, params=[ma_start_date]
    )

    conn.close()

    # Pre-index for fast lookup
    print(f"  Building indexes... ({len(ohlcv):,} OHLCV rows, {len(investor):,} investor rows)")

    # OHLCV by (date, ticker)
    ohlcv_by_date = {}
    for date, group in ohlcv.groupby("date"):
        ohlcv_by_date[date] = group.set_index("ticker")

    # Investor by (date, ticker)
    investor_by_date = {}
    for date, group in investor.groupby("date"):
        investor_by_date[date] = group.set_index("ticker")

    # Per-ticker sorted arrays for MA computation
    ticker_data = {}
    for ticker, group in ohlcv.groupby("ticker"):
        g = group.sort_values("date")
        ticker_data[ticker] = {
            "dates": g["date"].values,
            "close": g["close"].values,
            "open": g["open"].values,
            "high": g["high"].values,
            "low": g["low"].values,
            "volume": g["volume"].values,
        }

    # Date ordering for next-day lookup
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    print(f"  Preload complete. {len(ticker_data)} tickers indexed.")

    return {
        "test_dates": test_dates,
        "all_dates": all_dates,
        "ohlcv_by_date": ohlcv_by_date,
        "investor_by_date": investor_by_date,
        "ticker_data": ticker_data,
        "date_to_idx": date_to_idx,
    }


# ─────────────────────────────────────────────────────────────
# Fast Screener (no DB calls, pure memory)
# ─────────────────────────────────────────────────────────────

def fast_screen_day(
    target_date: str,
    data: dict,
    params: dict,
    strategy: str,
) -> list:
    """
    Screen one day with given parameters. Returns list of passing tickers
    with their entry context.
    """
    date_to_idx = data["date_to_idx"]
    all_dates = data["all_dates"]
    ohlcv_today = data["ohlcv_by_date"].get(target_date)
    investor_today = data["investor_by_date"].get(target_date)

    if ohlcv_today is None:
        return []

    # Previous 5 days for foreigner trend
    target_idx = date_to_idx.get(target_date)
    if target_idx is None or target_idx < 61:
        return []

    # Foreigner trend dates (last 5 trading days before target)
    trend_dates = all_dates[target_idx - 5:target_idx]

    results = []

    for ticker, td in data["ticker_data"].items():
        # Find target date index in this ticker's data
        date_positions = np.where(td["dates"] == target_date)[0]
        if len(date_positions) == 0:
            continue
        pos = date_positions[0]

        # Need 60 days of history
        if pos < 60:
            continue

        close_today = td["close"][pos]
        volume_today = td["volume"][pos]

        # Price filter
        if close_today < MIN_PRICE:
            continue

        # MAs
        ma5 = np.mean(td["close"][pos - 5:pos])
        ma20 = np.mean(td["close"][pos - 20:pos])
        ma60 = np.mean(td["close"][pos - 60:pos])

        # MA alignment
        if not (close_today > ma5 and close_today > ma20 and close_today > ma60):
            continue

        # Volume
        avg_vol_20 = np.mean(td["volume"][pos - 20:pos])
        if avg_vol_20 < MIN_AVG_VOLUME:
            continue
        volume_ratio = volume_today / avg_vol_20 if avg_vol_20 > 0 else 0

        if volume_ratio < params["volume_min"]:
            continue

        # Yesterday's high
        yesterday_high = td["high"][pos - 1]
        if yesterday_high <= 0:
            continue
        breakout_dist = (close_today - yesterday_high) / yesterday_high * 100

        if breakout_dist < params["breakout_min"]:
            continue
        if breakout_dist > params["breakout_max"]:
            continue

        # Investor checks
        if investor_today is not None and ticker in investor_today.index:
            fgn_net = investor_today.loc[ticker, "foreigner_net"]
            inst_net = investor_today.loc[ticker, "institution_net"]
        else:
            fgn_net = 0
            inst_net = 0

        # Foreigner today
        if params.get("require_foreigner_today", True) and fgn_net <= 0:
            continue

        # Strategy-specific filters
        if strategy == "hold":
            # Foreigner trend
            fgn_positive_days = 0
            for td_date in trend_dates:
                inv_day = data["investor_by_date"].get(td_date)
                if inv_day is not None and ticker in inv_day.index:
                    if inv_day.loc[ticker, "foreigner_net"] > 0:
                        fgn_positive_days += 1

            if fgn_positive_days < params.get("foreigner_trend_min", 3):
                continue

            # Institution
            if params.get("require_institution", False) and inst_net <= 0:
                continue

        results.append({
            "ticker": ticker,
            "pos": pos,
            "close": close_today,
            "volume_ratio": volume_ratio,
            "breakout_dist": breakout_dist,
        })

    return results


def get_next_day_returns(ticker: str, pos: int, data: dict) -> dict:
    """
    Get actual returns for a ticker after the setup day.
    Entry = next day's open. Returns open-to-high (DL) and open-to-close+N (hold).
    """
    td = data["ticker_data"].get(ticker)
    if td is None:
        return None

    next_pos = pos + 1
    if next_pos >= len(td["dates"]):
        return None

    entry = td["open"][next_pos]
    if entry <= 0:
        return None

    next_high = td["high"][next_pos]
    next_close = td["close"][next_pos]
    dl_return = (next_high - entry) / entry * 100
    dl_close_return = (next_close - entry) / entry * 100

    hold_returns = {}
    for period in HOLD_PERIODS:
        future_pos = next_pos + period
        if future_pos < len(td["dates"]):
            future_close = td["close"][future_pos]
            hold_returns[period] = (future_close - entry) / entry * 100
        else:
            hold_returns[period] = None

    return {
        "entry": entry,
        "dl_return": dl_return,
        "dl_close_return": dl_close_return,
        "hold_returns": hold_returns,
    }


# ─────────────────────────────────────────────────────────────
# Grid Search Engine
# ─────────────────────────────────────────────────────────────

def run_grid_search(
    data: dict,
    strategy: str,
    grid: dict,
) -> list:
    """
    Run exhaustive parameter sweep.
    Returns list of results sorted by expected value.
    """
    # Generate all parameter combinations
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    combos = list(itertools.product(*values))

    print(f"\n  Strategy: {strategy.upper()}")
    print(f"  Parameter combos: {len(combos)}")
    print(f"  Test dates: {len(data['test_dates'])}")
    print(f"  Total evaluations: {len(combos) * len(data['test_dates']):,}")

    results = []

    for ci, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        all_dl_returns = []
        all_dl_close_returns = []
        all_hold_returns = {p: [] for p in HOLD_PERIODS}
        total_candidates = 0

        for target_date in data["test_dates"]:
            candidates = fast_screen_day(target_date, data, params, strategy)
            total_candidates += len(candidates)

            for cand in candidates:
                ret = get_next_day_returns(cand["ticker"], cand["pos"], data)
                if ret is None:
                    continue

                all_dl_returns.append(ret["dl_return"])
                all_dl_close_returns.append(ret["dl_close_return"])
                for period in HOLD_PERIODS:
                    if ret["hold_returns"][period] is not None:
                        all_hold_returns[period].append(ret["hold_returns"][period])

        # Aggregate
        n = len(all_dl_returns)
        if n < 10:
            continue  # Skip combos with too few signals

        avg_per_day = total_candidates / len(data["test_dates"])

        if strategy == "day_leech":
            wins = sum(1 for r in all_dl_returns if r >= DAY_LEECH_TARGET_PCT)
            wr = wins / n * 100
            avg_ret = np.mean(all_dl_returns)
            avg_win = np.mean([r for r in all_dl_returns if r >= DAY_LEECH_TARGET_PCT]) if wins > 0 else 0
            avg_loss = np.mean([r for r in all_dl_returns if r < DAY_LEECH_TARGET_PCT]) if (n - wins) > 0 else 0
            median_ret = np.median(all_dl_returns)
            # Expected value: probability-weighted outcome
            ev = (wr / 100 * avg_win) + ((1 - wr / 100) * avg_loss) if n > 0 else 0

        else:  # hold
            # Use T+5 as primary metric
            returns = all_hold_returns[5]
            if len(returns) < 10:
                continue
            n = len(returns)
            wins = sum(1 for r in returns if r > 0)
            wr = wins / n * 100
            avg_ret = np.mean(returns)
            avg_win = np.mean([r for r in returns if r > 0]) if wins > 0 else 0
            avg_loss = np.mean([r for r in returns if r <= 0]) if (n - wins) > 0 else 0
            median_ret = np.median(returns)
            ev = avg_ret  # For holds, simple average is the EV

        rr = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        results.append({
            "params": params,
            "n_signals": n,
            "avg_per_day": round(avg_per_day, 1),
            "wr": round(wr, 1),
            "avg_ret": round(avg_ret, 2),
            "median_ret": round(median_ret, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "rr": round(rr, 2) if rr != float("inf") else 999,
            "ev": round(ev, 3),
            # DL close returns (realistic: buy open, sell close)
            "dl_close_wr": round(sum(1 for r in all_dl_close_returns if r > 0) / max(len(all_dl_close_returns), 1) * 100, 1),
            "dl_close_avg": round(np.mean(all_dl_close_returns), 2) if all_dl_close_returns else 0,
            "dl_close_median": round(np.median(all_dl_close_returns), 2) if all_dl_close_returns else 0,
            # Hold-specific: add T+1, T+3, T+10 for reference
            "hold_t1_wr": round(sum(1 for r in all_hold_returns[1] if r > 0) / max(len(all_hold_returns[1]), 1) * 100, 1),
            "hold_t3_wr": round(sum(1 for r in all_hold_returns[3] if r > 0) / max(len(all_hold_returns[3]), 1) * 100, 1),
            "hold_t5_wr": round(sum(1 for r in all_hold_returns[5] if r > 0) / max(len(all_hold_returns[5]), 1) * 100, 1),
            "hold_t10_wr": round(sum(1 for r in all_hold_returns[10] if r > 0) / max(len(all_hold_returns[10]), 1) * 100, 1),
        })

        if (ci + 1) % 20 == 0 or ci == 0:
            print(f"    [{ci+1}/{len(combos)}] {params} → n={n} WR={wr:.0f}% EV={ev:+.3f}")

    # Sort by EV descending
    results.sort(key=lambda x: x["ev"], reverse=True)

    return results


def print_results(results: list, strategy: str, top_n: int = 20):
    """Pretty print top results."""
    print(f"\n{'=' * 100}")
    print(f"  TOP {top_n} PARAMETER COMBOS — {strategy.upper()}")
    print(f"{'=' * 100}")

    if strategy == "day_leech":
        print(f"  {'Rank':<5} {'vol≥':>5} {'brk_min':>8} {'brk_max':>8} "
              f"{'Signals':>8} {'Avg/day':>8} "
              f"{'HiWR':>6} {'HiAvg':>8} {'R:R':>6} {'HiEV':>8} "
              f"{'│':>1} {'ClWR':>6} {'ClAvg':>8} {'ClMed':>8}")
        print(f"  {'─' * 105}")

        for i, r in enumerate(results[:top_n]):
            p = r["params"]
            print(f"  {i+1:<5} {p['volume_min']:>5.1f} {p['breakout_min']:>8.1f} {p['breakout_max']:>8.1f} "
                  f"{r['n_signals']:>8} {r['avg_per_day']:>8.1f} "
                  f"{r['wr']:>5.1f}% {r['avg_ret']:>+7.2f}% {r['rr']:>6.2f} {r['ev']:>+7.3f} "
                  f"{'│':>1} {r['dl_close_wr']:>5.1f}% {r['dl_close_avg']:>+7.2f}% {r['dl_close_median']:>+7.2f}%")
    else:
        print(f"  {'Rank':<5} {'vol≥':>5} {'brk_min':>8} {'brk_max':>8} {'fgn≥':>5} {'inst':>5} "
              f"{'Signals':>8} {'Avg/day':>8} {'T5 WR':>6} {'AvgRet':>8} {'MedRet':>8} "
              f"{'R:R':>6} {'EV':>8} {'T1':>5} {'T3':>5} {'T10':>5}")
        print(f"  {'─' * 115}")

        for i, r in enumerate(results[:top_n]):
            p = r["params"]
            inst_str = "Y" if p.get("require_institution", False) else "N"
            print(f"  {i+1:<5} {p['volume_min']:>5.1f} {p['breakout_min']:>8.1f} {p['breakout_max']:>8.1f} "
                  f"{p.get('foreigner_trend_min', '-'):>5} {inst_str:>5} "
                  f"{r['n_signals']:>8} {r['avg_per_day']:>8.1f} {r['wr']:>5.1f}% {r['avg_ret']:>+7.2f}% "
                  f"{r['median_ret']:>+7.2f}% {r['rr']:>6.2f} {r['ev']:>+7.3f} "
                  f"{r['hold_t1_wr']:>4.0f}% {r['hold_t3_wr']:>4.0f}% {r['hold_t10_wr']:>4.0f}%")

    # Also show bottom 5 (worst combos) for reference
    print(f"\n  --- WORST 5 ---")
    for i, r in enumerate(results[-5:]):
        p = r["params"]
        print(f"  {len(results)-4+i}. {p} → n={r['n_signals']} WR={r['wr']:.0f}% EV={r['ev']:+.3f}")


def save_results(results: list, strategy: str):
    """Save full results to CSV for analysis."""
    rows = []
    for r in results:
        row = {**r["params"], **{k: v for k, v in r.items() if k != "params"}}
        rows.append(row)
    df = pd.DataFrame(rows)
    path = f"data/grid_search_{strategy}.csv"
    df.to_csv(path, index=False)
    print(f"\n  Full results saved to {path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="일일티커 — Parameter Grid Search")
    parser.add_argument("--days", type=int, default=250,
                        help="Number of trading days to test (default: 250)")
    parser.add_argument("--strategy", type=str, default="both",
                        choices=["day_leech", "hold", "both"],
                        help="Which strategy to sweep")
    args = parser.parse_args()

    print("=" * 60)
    print("  일일티커 — Parameter Grid Search")
    print("=" * 60)

    data = preload_all_data(DB_PATH, args.days)

    if args.strategy in ("day_leech", "both"):
        dl_results = run_grid_search(data, "day_leech", DL_GRID)
        print_results(dl_results, "day_leech")
        save_results(dl_results, "day_leech")

    if args.strategy in ("hold", "both"):
        hold_results = run_grid_search(data, "hold", HOLD_GRID)
        print_results(hold_results, "hold")
        save_results(hold_results, "hold")

    print(f"\n{'=' * 60}")
    print(f"  Grid search complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()