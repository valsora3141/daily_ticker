"""
일일티커 — Portfolio Simulation

Simulates the actual product: each trading day, run the screener,
take the #1 recommendation, buy at next-day open, sell at T+5 close.

Only holds 1 position at a time. If already holding, skip new signals
until current position closes.

Usage:
    python scripts/simulate.py                    # last 250 days
    python scripts/simulate.py --days 500         # last 500 days
    python scripts/simulate.py --hold 3           # sell at T+3 instead of T+5
    python scripts/simulate.py --top 3            # buy top 3 (equal weight)
"""

import sys
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DB_PATH = "data/daily_cache.db"

# Screener params (grid search optimal)
MIN_PRICE = 1_000
MIN_AVG_VOLUME = 50_000
VOLUME_RATIO_MIN = 1.5
BREAKOUT_MIN_PCT = -2.0
BREAKOUT_MAX_PCT = 20.0
FOREIGNER_TREND_MIN = 4
ROUND_TRIP_FEE_PCT = 0.23  # buy + sell commissions + tax


def preload_data(db_path: str, n_days: int) -> dict:
    """Load all data into memory."""
    print(f"  Loading data...")
    conn = sqlite3.connect(db_path)

    all_dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT date FROM daily_ohlcv ORDER BY date"
    ).fetchall()]

    # Need 60 days before test window for MAs
    if len(all_dates) < n_days + 66:
        n_days = len(all_dates) - 66

    # Test dates: we need +6 days after the last test date for T+5 exit
    test_dates = all_dates[-(n_days + 6):-6]
    ma_start = all_dates[-(n_days + 66)]

    print(f"  Test window: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")

    ohlcv = pd.read_sql_query(
        "SELECT * FROM daily_ohlcv WHERE date >= ? ORDER BY ticker, date",
        conn, params=[ma_start]
    )

    investor = pd.read_sql_query(
        "SELECT * FROM daily_investor WHERE date >= ? ORDER BY ticker, date",
        conn, params=[ma_start]
    )

    conn.close()

    # Index
    investor_by_date = {}
    for date, group in investor.groupby("date"):
        investor_by_date[date] = group.set_index("ticker")

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

    date_to_idx_global = {d: i for i, d in enumerate(all_dates)}

    print(f"  {len(ticker_data)} tickers loaded.")

    return {
        "test_dates": test_dates,
        "all_dates": all_dates,
        "investor_by_date": investor_by_date,
        "ticker_data": ticker_data,
        "date_to_idx_global": date_to_idx_global,
    }


def screen_day(target_date: str, data: dict) -> list:
    """Screen one day, return candidates sorted by hold raw score."""
    all_dates = data["all_dates"]
    date_to_idx = data["date_to_idx_global"]
    investor_today = data["investor_by_date"].get(target_date)

    target_idx_global = date_to_idx.get(target_date)
    if target_idx_global is None or target_idx_global < 61:
        return []

    # Foreigner trend dates (last 5 trading days)
    trend_dates = all_dates[target_idx_global - 5:target_idx_global]

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

        # MAs
        ma5 = np.mean(td["close"][pos - 5:pos])
        ma20 = np.mean(td["close"][pos - 20:pos])
        ma60 = np.mean(td["close"][pos - 60:pos])

        if not (close > ma5 and close > ma20 and close > ma60):
            continue

        # Volume
        avg_vol = np.mean(td["volume"][pos - 20:pos])
        if avg_vol < MIN_AVG_VOLUME:
            continue
        vol_ratio = volume / avg_vol if avg_vol > 0 else 0
        if vol_ratio < VOLUME_RATIO_MIN:
            continue

        # Breakout
        yesterday_high = td["high"][pos - 1]
        if yesterday_high <= 0:
            continue
        brk_dist = (close - yesterday_high) / yesterday_high * 100
        if brk_dist < BREAKOUT_MIN_PCT or brk_dist > BREAKOUT_MAX_PCT:
            continue

        # Investor: foreigner today
        if investor_today is not None and ticker in investor_today.index:
            fgn = investor_today.loc[ticker, "foreigner_net"]
            inst = investor_today.loc[ticker, "institution_net"]
        else:
            fgn = 0
            inst = 0

        if fgn <= 0:
            continue
        if inst <= 0:
            continue

        # Foreigner trend
        fgn_positive = 0
        for td_date in trend_dates:
            inv_day = data["investor_by_date"].get(td_date)
            if inv_day is not None and ticker in inv_day.index:
                if inv_day.loc[ticker, "foreigner_net"] > 0:
                    fgn_positive += 1

        if fgn_positive < FOREIGNER_TREND_MIN:
            continue

        # Score (same as screener)
        score = 0.0
        score += min(fgn_positive * 0.8, 4.0)
        score += 1.5  # institution
        score += min(vol_ratio * 0.5, 2.0)
        if 0 <= brk_dist <= 5:
            score += 1.5
        elif -2 <= brk_dist < 0:
            score += 1.0
        elif 5 < brk_dist <= 15:
            score += 0.5

        candidates.append({
            "ticker": ticker,
            "pos": pos,
            "score": score,
            "close": close,
            "vol_ratio": vol_ratio,
            "fgn_trend": fgn_positive,
            "brk_dist": brk_dist,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def get_trade_result(ticker: str, pos: int, hold_days: int, data: dict) -> dict:
    """Get actual trade result: buy at pos+1 open, sell at pos+1+hold close."""
    td = data["ticker_data"].get(ticker)
    if td is None:
        return None

    entry_pos = pos + 1
    exit_pos = entry_pos + hold_days

    if exit_pos >= len(td["dates"]):
        return None

    entry_price = td["open"][entry_pos]
    exit_price = td["close"][exit_pos]

    if entry_price <= 0:
        return None

    # Max adverse (worst drawdown during hold)
    lows_during = td["low"][entry_pos:exit_pos + 1]
    max_adverse = (min(lows_during) - entry_price) / entry_price * 100

    # Max favorable
    highs_during = td["high"][entry_pos:exit_pos + 1]
    max_favorable = (max(highs_during) - entry_price) / entry_price * 100

    gross_return = (exit_price - entry_price) / entry_price * 100
    net_return = gross_return - ROUND_TRIP_FEE_PCT

    return {
        "entry_date": td["dates"][entry_pos],
        "exit_date": td["dates"][exit_pos],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_return": gross_return,
        "net_return": net_return,
        "max_adverse": max_adverse,
        "max_favorable": max_favorable,
    }


def simulate(data: dict, hold_days: int = 5, top_n: int = 1):
    """Run the full simulation."""
    print(f"\n  Simulation: top {top_n}, hold T+{hold_days}, {len(data['test_dates'])} days")
    print(f"  Fee per round-trip: {ROUND_TRIP_FEE_PCT}%")

    capital = 10_000_000  # 10M KRW
    equity = capital
    position_count = 0
    positions = []  # Active positions: {ticker, entry_date, exit_date, ...}
    trades = []     # Completed trades
    equity_curve = []
    daily_log = []

    for day_idx, target_date in enumerate(data["test_dates"]):
        # Check if any positions close today
        closed_today = []
        still_open = []
        for p in positions:
            if p["exit_date"] <= target_date:
                # Close this position
                pnl = p["allocation"] * (p["net_return"] / 100)
                equity += p["allocation"] + pnl
                position_count -= 1
                p["pnl"] = pnl
                trades.append(p)
                closed_today.append(p)
            else:
                still_open.append(p)
        positions = still_open

        # Screen today
        candidates = screen_day(target_date, data)

        # Open new positions if we have capacity
        new_positions = 0
        if position_count < top_n and candidates:
            slots_available = top_n - position_count
            for cand in candidates[:slots_available]:
                result = get_trade_result(cand["ticker"], cand["pos"], hold_days, data)
                if result is None:
                    continue

                allocation = equity / (top_n - position_count)  # Equal weight remaining
                allocation = min(allocation, equity)  # Can't exceed available cash

                if allocation <= 0:
                    continue

                equity -= allocation
                position_count += 1
                new_positions += 1

                pos_record = {
                    **result,
                    "ticker": cand["ticker"],
                    "signal_date": target_date,
                    "score": cand["score"],
                    "allocation": allocation,
                }
                positions.append(pos_record)

        # Mark-to-market for equity curve
        # (approximate: use entry allocation + estimated P&L)
        unrealized = 0
        for p in positions:
            td = data["ticker_data"].get(p["ticker"])
            if td is not None:
                date_positions = np.where(td["dates"] == target_date)[0]
                if len(date_positions) > 0:
                    current_price = td["close"][date_positions[0]]
                    unrealized_pct = (current_price - p["entry_price"]) / p["entry_price"] * 100
                    unrealized += p["allocation"] * (unrealized_pct / 100)

        total_equity = equity + sum(p["allocation"] for p in positions) + unrealized

        equity_curve.append({
            "date": target_date,
            "total_equity": total_equity,
            "cash": equity,
            "positions": position_count,
            "new_trades": new_positions,
            "closed_trades": len(closed_today),
        })

    # Close any remaining positions at last available price
    for p in positions:
        pnl = p["allocation"] * (p["net_return"] / 100)
        equity += p["allocation"] + pnl
        p["pnl"] = pnl
        trades.append(p)

    return trades, equity_curve, capital


def print_results(trades: list, equity_curve: list, initial_capital: float):
    """Print simulation results."""
    if not trades:
        print("\n  No trades executed.")
        return

    print(f"\n{'=' * 70}")
    print(f"  SIMULATION RESULTS")
    print(f"{'=' * 70}")

    # Overview
    final_equity = equity_curve[-1]["total_equity"]
    total_return = (final_equity - initial_capital) / initial_capital * 100
    n_trades = len(trades)
    wins = [t for t in trades if t["net_return"] > 0]
    losses = [t for t in trades if t["net_return"] <= 0]
    wr = len(wins) / n_trades * 100 if n_trades > 0 else 0

    print(f"\n  Capital: {initial_capital:,.0f}원 → {final_equity:,.0f}원")
    print(f"  Total return: {total_return:+.2f}%")
    print(f"  Trades: {n_trades} (W:{len(wins)} L:{len(losses)} WR:{wr:.1f}%)")

    # Days
    trading_days = len(equity_curve)
    days_with_position = sum(1 for e in equity_curve if e["positions"] > 0)
    print(f"  Trading days: {trading_days}")
    print(f"  Days with position: {days_with_position} ({days_with_position/trading_days*100:.0f}%)")

    # Returns
    net_returns = [t["net_return"] for t in trades]
    print(f"\n  Avg return/trade: {np.mean(net_returns):+.2f}%")
    print(f"  Median return/trade: {np.median(net_returns):+.2f}%")
    if wins:
        print(f"  Avg win: {np.mean([t['net_return'] for t in wins]):+.2f}%")
    if losses:
        print(f"  Avg loss: {np.mean([t['net_return'] for t in losses]):+.2f}%")

    # Risk
    max_favorable = [t["max_favorable"] for t in trades]
    max_adverse = [t["max_adverse"] for t in trades]
    print(f"\n  Avg max favorable (during hold): {np.mean(max_favorable):+.2f}%")
    print(f"  Avg max adverse (during hold): {np.mean(max_adverse):+.2f}%")

    # Drawdown
    equities = [e["total_equity"] for e in equity_curve]
    peak = equities[0]
    max_dd = 0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd
    print(f"  Max drawdown: {max_dd:.2f}%")

    # Annualized
    if trading_days > 20:
        annual_factor = 250 / trading_days
        annualized = ((final_equity / initial_capital) ** annual_factor - 1) * 100
        print(f"  Annualized return: {annualized:+.1f}%")

        # Sharpe (daily returns)
        daily_returns = []
        for i in range(1, len(equities)):
            dr = (equities[i] - equities[i-1]) / equities[i-1]
            daily_returns.append(dr)
        if daily_returns:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(250) if np.std(daily_returns) > 0 else 0
            print(f"  Sharpe ratio: {sharpe:.2f}")

    # Monthly breakdown
    print(f"\n  Monthly breakdown:")
    print(f"  {'Month':<10} {'Trades':>6} {'WR':>6} {'P&L':>12} {'Return':>8}")
    print(f"  {'─' * 46}")

    by_month = defaultdict(list)
    for t in trades:
        month = t["entry_date"][:7]
        by_month[month].append(t)

    for month in sorted(by_month.keys()):
        mt = by_month[month]
        m_wins = sum(1 for t in mt if t["net_return"] > 0)
        m_wr = m_wins / len(mt) * 100
        m_pnl = sum(t["pnl"] for t in mt)
        m_ret = sum(t["net_return"] for t in mt) / len(mt)
        print(f"  {month:<10} {len(mt):>6} {m_wr:>5.0f}% {m_pnl:>+11,.0f}원 {m_ret:>+7.2f}%")

    # Worst trades
    sorted_trades = sorted(trades, key=lambda t: t["net_return"])
    print(f"\n  Worst 5 trades:")
    for t in sorted_trades[:5]:
        print(f"    {t['signal_date']} {t['ticker']} {t['net_return']:+.2f}% "
              f"(max adverse: {t['max_adverse']:+.1f}%)")

    # Best trades
    print(f"\n  Best 5 trades:")
    for t in sorted_trades[-5:]:
        print(f"    {t['signal_date']} {t['ticker']} {t['net_return']:+.2f}% "
              f"(max favorable: {t['max_favorable']:+.1f}%)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="일일티커 — Portfolio Simulation")
    parser.add_argument("--days", type=int, default=250,
                        help="Number of trading days to simulate (default: 250)")
    parser.add_argument("--hold", type=int, default=5,
                        help="Hold period in trading days (default: 5)")
    parser.add_argument("--top", type=int, default=1,
                        help="Number of positions (default: 1)")
    args = parser.parse_args()

    print("=" * 60)
    print("  일일티커 — Portfolio Simulation")
    print("=" * 60)

    data = preload_data(DB_PATH, args.days)
    trades, equity_curve, capital = simulate(data, hold_days=args.hold, top_n=args.top)
    print_results(trades, equity_curve, capital)

    # Save equity curve
    df = pd.DataFrame(equity_curve)
    df.to_csv("data/equity_curve.csv", index=False)
    print(f"\n  Equity curve saved to data/equity_curve.csv")

    # Save trades
    trade_records = []
    for t in trades:
        trade_records.append({
            "signal_date": t["signal_date"],
            "ticker": t["ticker"],
            "entry_date": t["entry_date"],
            "exit_date": t["exit_date"],
            "entry_price": t["entry_price"],
            "exit_price": t["exit_price"],
            "gross_return": round(t["gross_return"], 2),
            "net_return": round(t["net_return"], 2),
            "max_favorable": round(t["max_favorable"], 2),
            "max_adverse": round(t["max_adverse"], 2),
            "score": round(t["score"], 2),
            "allocation": round(t["allocation"], 0),
            "pnl": round(t["pnl"], 0),
        })
    pd.DataFrame(trade_records).to_csv("data/simulation_trades.csv", index=False)
    print(f"  Trades saved to data/simulation_trades.csv")


if __name__ == "__main__":
    main()