"""
일일티커 — Walk-Forward Analysis

5 folds of train/test splits:
  Fold 1: Train 2020-2021, Test 2022
  Fold 2: Train 2020-2022, Test 2023
  Fold 3: Train 2020-2023, Test 2024
  Fold 4: Train 2020-2024, Test 2025
  Fold 5: Train 2020-2025, Test 2026

For each fold:
  1. Grid search optimal hold params on train set
  2. Simulate top-1 pick on test set with those params
  3. Report WR, return, Sharpe, drawdown

Usage:
    python scripts/walk_forward.py
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

DB_PATH = "data/daily_cache.db"

# ─────────────────────────────────────────────────────────────
# Fixed grid (same for all folds)
# ─────────────────────────────────────────────────────────────

HOLD_GRID = {
    "volume_min": [0.8, 1.0, 1.3, 1.5],
    "breakout_min": [-3.0, -2.0, -1.0],
    "breakout_max": [8.0, 12.0, 15.0, 20.0],
    "foreigner_trend_min": [2, 3, 4],
    "require_institution": [True, False],
}

# Fixed constants
MIN_PRICE = 1_000
MIN_AVG_VOLUME = 50_000
HOLD_DAYS = 5
ROUND_TRIP_FEE_PCT = 0.23
HOLD_PERIODS = [1, 3, 5, 10]

# Folds
FOLDS = [
    {"name": "Fold 1", "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
    {"name": "Fold 2", "train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31"},
    {"name": "Fold 3", "train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"},
    {"name": "Fold 4", "train_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2025-12-31"},
    {"name": "Fold 5", "train_end": "2025-12-31", "test_start": "2026-01-01", "test_end": "2026-12-31"},
]


# ─────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────

def load_all_data(db_path: str) -> dict:
    """Load everything into memory once."""
    print("  Loading all data into memory...")
    conn = sqlite3.connect(db_path)

    all_dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT date FROM daily_ohlcv ORDER BY date"
    ).fetchall()]

    ohlcv = pd.read_sql_query(
        "SELECT * FROM daily_ohlcv ORDER BY ticker, date", conn
    )
    investor = pd.read_sql_query(
        "SELECT * FROM daily_investor ORDER BY ticker, date", conn
    )
    conn.close()

    # Index investor by date
    investor_by_date = {}
    for date, group in investor.groupby("date"):
        investor_by_date[date] = group.set_index("ticker")

    # Per-ticker arrays
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

    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    print(f"  Loaded {len(ohlcv):,} OHLCV rows, {len(investor):,} investor rows")
    print(f"  {len(ticker_data)} tickers, {len(all_dates)} trading days")
    print(f"  Date range: {all_dates[0]} to {all_dates[-1]}")

    return {
        "all_dates": all_dates,
        "investor_by_date": investor_by_date,
        "ticker_data": ticker_data,
        "date_to_idx": date_to_idx,
    }


def get_dates_in_range(all_dates: list, start: str, end: str) -> list:
    """Filter trading dates within a range."""
    return [d for d in all_dates if start <= d <= end]


# ─────────────────────────────────────────────────────────────
# Screening (parameterized)
# ─────────────────────────────────────────────────────────────

def screen_day(target_date: str, data: dict, params: dict) -> list:
    """Screen one day with given params. Returns candidates sorted by score."""
    all_dates = data["all_dates"]
    date_to_idx = data["date_to_idx"]
    investor_today = data["investor_by_date"].get(target_date)

    target_idx = date_to_idx.get(target_date)
    if target_idx is None or target_idx < 61:
        return []

    trend_dates = all_dates[target_idx - 5:target_idx]

    candidates = []

    for ticker, td in data["ticker_data"].items():
        date_positions = np.where(td["dates"] == target_date)[0]
        if len(date_positions) == 0:
            continue
        pos = date_positions[0]
        if pos < 60:
            continue

        close = td["close"][pos]
        volume = td["volume"][pos]

        if close < MIN_PRICE:
            continue

        ma5 = np.mean(td["close"][pos - 5:pos])
        ma20 = np.mean(td["close"][pos - 20:pos])
        ma60 = np.mean(td["close"][pos - 60:pos])

        if not (close > ma5 and close > ma20 and close > ma60):
            continue

        avg_vol = np.mean(td["volume"][pos - 20:pos])
        if avg_vol < MIN_AVG_VOLUME:
            continue
        vol_ratio = volume / avg_vol if avg_vol > 0 else 0
        if vol_ratio < params["volume_min"]:
            continue

        yesterday_high = td["high"][pos - 1]
        if yesterday_high <= 0:
            continue
        brk_dist = (close - yesterday_high) / yesterday_high * 100
        if brk_dist < params["breakout_min"] or brk_dist > params["breakout_max"]:
            continue

        if investor_today is not None and ticker in investor_today.index:
            fgn = investor_today.loc[ticker, "foreigner_net"]
            inst = investor_today.loc[ticker, "institution_net"]
        else:
            fgn = 0
            inst = 0

        if fgn <= 0:
            continue
        if params.get("require_institution", False) and inst <= 0:
            continue

        fgn_positive = 0
        for td_date in trend_dates:
            inv_day = data["investor_by_date"].get(td_date)
            if inv_day is not None and ticker in inv_day.index:
                if inv_day.loc[ticker, "foreigner_net"] > 0:
                    fgn_positive += 1

        if fgn_positive < params.get("foreigner_trend_min", 3):
            continue

        # Score
        score = 0.0
        score += min(fgn_positive * 0.8, 4.0)
        if inst > 0:
            score += 1.5
        score += min(vol_ratio * 0.5, 2.0)
        if 0 <= brk_dist <= 5:
            score += 1.5
        elif -2 <= brk_dist < 0:
            score += 1.0
        elif 5 < brk_dist <= 15:
            score += 0.5
        if fgn > 0:
            score += 0.5

        candidates.append({
            "ticker": ticker,
            "pos": pos,
            "score": score,
            "close": close,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


# ─────────────────────────────────────────────────────────────
# Grid Search (on train set) — Parallelized
# ─────────────────────────────────────────────────────────────

# Global reference for worker processes
_worker_data = None
_worker_eval_dates = None


def _init_worker(data, eval_dates):
    """Initialize worker process with shared data."""
    global _worker_data, _worker_eval_dates
    _worker_data = data
    _worker_eval_dates = eval_dates


def _eval_combo(args):
    """Evaluate a single parameter combo. Runs in worker process."""
    keys, combo = args
    params = dict(zip(keys, combo))

    returns = []

    for target_date in _worker_eval_dates:
        candidates = screen_day(target_date, _worker_data, params)
        if not candidates:
            continue

        cand = candidates[0]
        ticker = cand["ticker"]
        pos = cand["pos"]
        td = _worker_data["ticker_data"].get(ticker)
        if td is None:
            continue

        entry_pos = pos + 1
        exit_pos = entry_pos + HOLD_DAYS

        if exit_pos >= len(td["dates"]):
            continue

        entry = td["open"][entry_pos]
        exit_price = td["close"][exit_pos]
        if entry <= 0:
            continue

        ret = (exit_price - entry) / entry * 100 - ROUND_TRIP_FEE_PCT
        returns.append(ret)

    if len(returns) < 10:
        return None

    wr = sum(1 for r in returns if r > 0) / len(returns) * 100
    ev = np.mean(returns)
    median = np.median(returns)

    return {
        "params": params,
        "n": len(returns),
        "wr": round(wr, 1),
        "ev": round(ev, 3),
        "median": round(median, 2),
    }


def grid_search_train(data: dict, train_dates: list, n_workers: int = 10) -> dict:
    """
    Grid search on training dates using multiprocessing.
    Each combo evaluated in parallel across worker processes.
    """
    import multiprocessing as mp

    keys = sorted(HOLD_GRID.keys())
    values = [HOLD_GRID[k] for k in keys]
    combos = list(itertools.product(*values))
    eval_dates = train_dates[60:]

    print(f"    Grid: {len(combos)} combos × {len(eval_dates)} days × {n_workers} workers")

    # Prepare args
    args_list = [(keys, combo) for combo in combos]

    # Run parallel
    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(data, eval_dates),
    ) as pool:
        results_raw = []
        for i, result in enumerate(pool.imap_unordered(_eval_combo, args_list)):
            if result is not None:
                results_raw.append(result)
            if (i + 1) % 50 == 0:
                best_so_far = max((r["ev"] for r in results_raw), default=-999)
                print(f"      [{i+1}/{len(combos)}] completed, best EV so far: {best_so_far:+.3f}")

    if not results_raw:
        return {"best_params": None, "best_ev": -999, "top5": [], "total_combos_tested": 0}

    results_raw.sort(key=lambda x: x["ev"], reverse=True)

    return {
        "best_params": results_raw[0]["params"],
        "best_ev": results_raw[0]["ev"],
        "top5": results_raw[:5],
        "total_combos_tested": len(results_raw),
    }


# ─────────────────────────────────────────────────────────────
# Simulation (on test set)
# ─────────────────────────────────────────────────────────────

def simulate_test(data: dict, test_dates: list, params: dict) -> dict:
    """
    Simulate top-1 pick on test dates.
    Buy at next-day open, sell at T+5 close. 1 position at a time.
    """
    capital = 10_000_000
    equity = capital
    positions = []
    trades = []
    equity_curve = []

    for target_date in test_dates:
        # Close expired positions
        still_open = []
        for p in positions:
            if p["exit_date"] <= target_date:
                pnl = p["allocation"] * (p["net_return"] / 100)
                equity += p["allocation"] + pnl
                p["pnl"] = pnl
                trades.append(p)
            else:
                still_open.append(p)
        positions = still_open

        # Screen and open new position if empty
        if len(positions) == 0:
            candidates = screen_day(target_date, data, params)
            if candidates:
                cand = candidates[0]
                ticker = cand["ticker"]
                pos = cand["pos"]
                td = data["ticker_data"].get(ticker)

                if td is not None:
                    entry_pos = pos + 1
                    exit_pos = entry_pos + HOLD_DAYS

                    if exit_pos < len(td["dates"]):
                        entry_price = td["open"][entry_pos]
                        exit_price = td["close"][exit_pos]

                        if entry_price > 0:
                            gross_ret = (exit_price - entry_price) / entry_price * 100
                            net_ret = gross_ret - ROUND_TRIP_FEE_PCT

                            lows = td["low"][entry_pos:exit_pos + 1]
                            highs = td["high"][entry_pos:exit_pos + 1]
                            max_adv = (min(lows) - entry_price) / entry_price * 100
                            max_fav = (max(highs) - entry_price) / entry_price * 100

                            allocation = equity
                            equity = 0

                            positions.append({
                                "ticker": ticker,
                                "signal_date": target_date,
                                "entry_date": td["dates"][entry_pos],
                                "exit_date": td["dates"][exit_pos],
                                "entry_price": entry_price,
                                "exit_price": exit_price,
                                "gross_return": gross_ret,
                                "net_return": net_ret,
                                "max_adverse": max_adv,
                                "max_favorable": max_fav,
                                "allocation": allocation,
                                "score": cand["score"],
                            })

        # Mark-to-market
        unrealized = 0
        for p in positions:
            td = data["ticker_data"].get(p["ticker"])
            if td is not None:
                dpos = np.where(td["dates"] == target_date)[0]
                if len(dpos) > 0:
                    cur = td["close"][dpos[0]]
                    unrealized += p["allocation"] * ((cur - p["entry_price"]) / p["entry_price"])

        total_eq = equity + sum(p["allocation"] for p in positions) + unrealized
        equity_curve.append({"date": target_date, "equity": total_eq})

    # Close remaining
    for p in positions:
        pnl = p["allocation"] * (p["net_return"] / 100)
        equity += p["allocation"] + pnl
        p["pnl"] = pnl
        trades.append(p)

    # Compute stats
    if not trades:
        return {"n_trades": 0, "wr": 0, "total_return": 0, "sharpe": 0,
                "max_dd": 0, "avg_return": 0, "median_return": 0,
                "trades": trades, "equity_curve": equity_curve}

    final_eq = equity if not equity_curve else equity_curve[-1]["equity"]
    total_ret = (final_eq - capital) / capital * 100
    net_returns = [t["net_return"] for t in trades]
    wins = sum(1 for r in net_returns if r > 0)
    wr = wins / len(trades) * 100

    # Drawdown
    eqs = [e["equity"] for e in equity_curve]
    peak = eqs[0] if eqs else capital
    max_dd = 0
    for eq in eqs:
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd

    # Sharpe
    sharpe = 0
    if len(eqs) > 1:
        daily_rets = [(eqs[i] - eqs[i-1]) / eqs[i-1] for i in range(1, len(eqs))]
        if np.std(daily_rets) > 0:
            sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(250)

    # Annualized
    n_days = len(equity_curve)
    if n_days > 20:
        annual = ((final_eq / capital) ** (250 / n_days) - 1) * 100
    else:
        annual = total_ret

    return {
        "n_trades": len(trades),
        "wr": round(wr, 1),
        "total_return": round(total_ret, 2),
        "annualized": round(annual, 1),
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd, 2),
        "avg_return": round(np.mean(net_returns), 2),
        "median_return": round(np.median(net_returns), 2),
        "avg_win": round(np.mean([r for r in net_returns if r > 0]), 2) if wins > 0 else 0,
        "avg_loss": round(np.mean([r for r in net_returns if r <= 0]), 2) if (len(trades) - wins) > 0 else 0,
        "test_days": n_days,
        "trades": trades,
        "equity_curve": equity_curve,
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  일일티커 — Walk-Forward Analysis")
    print("=" * 70)

    data = load_all_data(DB_PATH)
    all_dates = data["all_dates"]

    fold_results = []

    for fold in FOLDS:
        print(f"\n{'─' * 70}")
        print(f"  {fold['name']}: Train ≤{fold['train_end']} | Test {fold['test_start']}~{fold['test_end']}")
        print(f"{'─' * 70}")

        train_dates = get_dates_in_range(all_dates, "2020-01-01", fold["train_end"])
        test_dates = get_dates_in_range(all_dates, fold["test_start"], fold["test_end"])

        if len(test_dates) < 5:
            print(f"    SKIP: only {len(test_dates)} test dates")
            fold_results.append({"fold": fold["name"], "skip": True})
            continue

        print(f"    Train: {len(train_dates)} days | Test: {len(test_dates)} days")

        # Grid search on train
        print(f"\n    [1/2] Grid searching on train set...")
        gs = grid_search_train(data, train_dates)

        if gs["best_params"] is None:
            print(f"    SKIP: no valid params found")
            fold_results.append({"fold": fold["name"], "skip": True})
            continue

        bp = gs["best_params"]
        print(f"\n    Best params: vol≥{bp['volume_min']} brk=[{bp['breakout_min']},{bp['breakout_max']}] "
              f"fgn≥{bp['foreigner_trend_min']} inst={'Y' if bp['require_institution'] else 'N'}")
        print(f"    Train EV: {gs['best_ev']:+.3f}%")
        print(f"    Top 5 combos on train:")
        for r in gs["top5"]:
            p = r["params"]
            print(f"      vol≥{p['volume_min']} brk=[{p['breakout_min']},{p['breakout_max']}] "
                  f"fgn≥{p['foreigner_trend_min']} inst={'Y' if p['require_institution'] else 'N'} "
                  f"→ n={r['n']} WR={r['wr']}% EV={r['ev']:+.3f}")

        # Simulate on test
        print(f"\n    [2/2] Simulating on test set ({fold['test_start']} ~ {fold['test_end']})...")
        sim = simulate_test(data, test_dates, gs["best_params"])

        print(f"\n    Results:")
        print(f"      Trades: {sim['n_trades']} | WR: {sim['wr']}%")
        print(f"      Total return: {sim['total_return']:+.2f}% | Annualized: {sim['annualized']:+.1f}%")
        print(f"      Avg/trade: {sim['avg_return']:+.2f}% | Median: {sim['median_return']:+.2f}%")
        if sim['n_trades'] > 0:
            print(f"      Avg win: {sim['avg_win']:+.2f}% | Avg loss: {sim['avg_loss']:+.2f}%")
        print(f"      Sharpe: {sim['sharpe']} | Max DD: {sim['max_dd']:.2f}%")

        fold_results.append({
            "fold": fold["name"],
            "skip": False,
            "train_end": fold["train_end"],
            "test_period": f"{fold['test_start'][:4]}",
            "best_params": gs["best_params"],
            "train_ev": gs["best_ev"],
            **sim,
        })

    # ─────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────

    print(f"\n\n{'=' * 90}")
    print(f"  WALK-FORWARD SUMMARY")
    print(f"{'=' * 90}")

    active_folds = [f for f in fold_results if not f.get("skip", False)]

    if not active_folds:
        print("  No valid folds.")
        return

    # Summary table
    print(f"\n  {'Fold':<8} {'Test':>5} {'Trades':>7} {'WR':>6} {'TotRet':>8} {'Annual':>8} "
          f"{'Avg/T':>7} {'Med/T':>7} {'Sharpe':>7} {'MaxDD':>7} "
          f"{'vol':>4} {'brk':>10} {'fgn':>4} {'inst':>5}")
    print(f"  {'─' * 105}")

    for f in active_folds:
        bp = f["best_params"]
        print(f"  {f['fold']:<8} {f['test_period']:>5} {f['n_trades']:>7} {f['wr']:>5.1f}% "
              f"{f['total_return']:>+7.2f}% {f['annualized']:>+7.1f}% "
              f"{f['avg_return']:>+6.2f}% {f['median_return']:>+6.2f}% "
              f"{f['sharpe']:>7.2f} {f['max_dd']:>6.2f}% "
              f"{bp['volume_min']:>4.1f} [{bp['breakout_min']},{bp['breakout_max']}] "
              f"{bp['foreigner_trend_min']:>3} {'Y' if bp['require_institution'] else 'N':>5}")

    # Aggregate out-of-sample stats
    all_trades = []
    for f in active_folds:
        all_trades.extend(f.get("trades", []))

    if all_trades:
        all_net = [t["net_return"] for t in all_trades]
        all_wins = sum(1 for r in all_net if r > 0)
        all_wr = all_wins / len(all_trades) * 100

        print(f"\n  AGGREGATE OUT-OF-SAMPLE:")
        print(f"    Total trades: {len(all_trades)}")
        print(f"    WR: {all_wr:.1f}%")
        print(f"    Avg return/trade: {np.mean(all_net):+.2f}%")
        print(f"    Median return/trade: {np.median(all_net):+.2f}%")
        if all_wins > 0:
            print(f"    Avg win: {np.mean([r for r in all_net if r > 0]):+.2f}%")
        if len(all_trades) - all_wins > 0:
            print(f"    Avg loss: {np.mean([r for r in all_net if r <= 0]):+.2f}%")

    # Param stability
    print(f"\n  PARAMETER STABILITY:")
    param_keys = ["volume_min", "breakout_min", "breakout_max", "foreigner_trend_min", "require_institution"]
    for key in param_keys:
        values = [f["best_params"][key] for f in active_folds]
        unique = set(values)
        if len(unique) == 1:
            stability = "STABLE ✓"
        elif len(unique) <= 2:
            stability = "MOSTLY STABLE"
        else:
            stability = "UNSTABLE ✗"
        print(f"    {key:<25} values={values}  → {stability}")

    # Save all trades
    trade_records = []
    for f in active_folds:
        for t in f.get("trades", []):
            trade_records.append({
                "fold": f["fold"],
                "test_year": f["test_period"],
                "signal_date": t["signal_date"],
                "ticker": t["ticker"],
                "entry_date": t["entry_date"],
                "exit_date": t["exit_date"],
                "entry_price": t["entry_price"],
                "exit_price": t["exit_price"],
                "net_return": round(t["net_return"], 2),
                "max_favorable": round(t["max_favorable"], 2),
                "max_adverse": round(t["max_adverse"], 2),
            })
    if trade_records:
        pd.DataFrame(trade_records).to_csv("data/walk_forward_trades.csv", index=False)
        print(f"\n  All trades saved to data/walk_forward_trades.csv")


if __name__ == "__main__":
    main()