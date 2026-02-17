"""
일일티커 — Data Fetcher

Fetches daily OHLCV + investor breakdown from KRX via pykrx,
caches in SQLite. Extended from SilentOverlord's KOSPICollector.

Usage:
    from scanner.data_fetcher import DailyDataFetcher

    fetcher = DailyDataFetcher("data/daily_cache.db")
    fetcher.collect_all("20260101", "20260217")
    fetcher.update()  # fetch only missing dates up to today
"""

import sqlite3
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, List

try:
    from pykrx import stock as pykrx_stock
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False


class DailyDataFetcher:
    """
    Fetches and caches KOSPI daily data:
      - OHLCV (from SilentOverlord's collector, same schema)
      - Investor breakdown (foreigner, institution, individual net buy)
    """

    def __init__(self, db_path: str = "data/daily_cache.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")

        # === OHLCV (same as SilentOverlord) ===
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_ohlcv (
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                trade_value REAL,
                change_pct REAL,
                market_cap REAL,
                PRIMARY KEY (date, ticker)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker
            ON daily_ohlcv (ticker, date)
        """)

        # === Investor breakdown (foreigner / institution / individual) ===
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_investor (
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                foreigner_net REAL,
                institution_net REAL,
                individual_net REAL,
                PRIMARY KEY (date, ticker)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_investor_ticker
            ON daily_investor (ticker, date)
        """)

        # === Collection log ===
        conn.execute("""
            CREATE TABLE IF NOT EXISTS collection_log (
                date TEXT NOT NULL,
                data_type TEXT NOT NULL,
                tickers_count INTEGER,
                collected_at TEXT,
                PRIMARY KEY (date, data_type)
            )
        """)

        conn.commit()
        conn.close()

    # ─────────────────────────────────────────────────────────
    # Collection
    # ─────────────────────────────────────────────────────────

    def update(self, market: str = "KOSPI"):
        """Fetch all missing dates up to today."""
        if not PYKRX_AVAILABLE:
            raise ImportError("pykrx required: pip install pykrx")

        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT MAX(date) FROM collection_log WHERE data_type = 'ohlcv'"
        ).fetchone()
        conn.close()

        if row[0]:
            start = (pd.Timestamp(row[0]) + pd.Timedelta(days=1)).strftime("%Y%m%d")
        else:
            # Default: 90 calendar days back (~60 trading days for MA60)
            start = (pd.Timestamp.now() - pd.Timedelta(days=90)).strftime("%Y%m%d")

        end = datetime.now().strftime("%Y%m%d")
        print(f"Updating from {start} to {end}...")
        self.collect_all(start, end, market=market)

    def collect_all(
        self,
        start_date: str,
        end_date: str,
        sleep_sec: float = 0.1,
        market: str = "KOSPI",
    ):
        """Fetch OHLCV + investor data for all dates in range."""
        if not PYKRX_AVAILABLE:
            raise ImportError("pykrx required: pip install pykrx")

        trading_days = self._get_trading_days(start_date, end_date)
        collected_ohlcv = self._get_collected_dates("ohlcv")
        collected_investor = self._get_collected_dates("investor")

        remaining_ohlcv = [d for d in trading_days if d not in collected_ohlcv]
        remaining_investor = [d for d in trading_days if d not in collected_investor]

        print(f"Trading days in range: {len(trading_days)}")
        print(f"OHLCV remaining: {len(remaining_ohlcv)}")
        print(f"Investor remaining: {len(remaining_investor)}")

        if remaining_ohlcv:
            print(f"\n--- Collecting OHLCV ---")
            self._collect_ohlcv(remaining_ohlcv, market, sleep_sec)

        if remaining_investor:
            print(f"\n--- Collecting Investor Breakdown ---")
            self._collect_investor(remaining_investor, market, sleep_sec)

        print(f"\nDone!")

    def _collect_ohlcv(self, dates: list, market: str, sleep_sec: float):
        conn = sqlite3.connect(self.db_path)
        errors = []

        for i, day in enumerate(dates):
            day_str = day.replace("-", "")
            try:
                df = pykrx_stock.get_market_ohlcv(day_str, market=market)

                if df.empty:
                    self._log_collection(conn, day, "ohlcv", 0)
                    continue

                df = df[df["거래량"] > 0]
                if df.empty:
                    self._log_collection(conn, day, "ohlcv", 0)
                    continue

                records = []
                for ticker, row in df.iterrows():
                    records.append((
                        day, ticker,
                        row["시가"], row["고가"], row["저가"], row["종가"],
                        int(row["거래량"]),
                        row.get("거래대금", 0),
                        row.get("등락률", 0),
                        row.get("시가총액", 0),
                    ))

                conn.executemany("""
                    INSERT OR REPLACE INTO daily_ohlcv
                    (date, ticker, open, high, low, close, volume,
                     trade_value, change_pct, market_cap)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, records)

                self._log_collection(conn, day, "ohlcv", len(records))

                if (i + 1) % 20 == 0 or i == 0:
                    remaining_est = (len(dates) - i - 1) * (0.8 + sleep_sec) / 60
                    print(f"  [{i+1}/{len(dates)}] {day} — {len(records)} tickers "
                          f"| ~{remaining_est:.1f} min left")

                time.sleep(sleep_sec)

            except Exception as e:
                errors.append((day, str(e)))
                print(f"  [{i+1}/{len(dates)}] {day} — ERROR: {e}")
                time.sleep(2)

        conn.close()
        if errors:
            print(f"  OHLCV errors: {len(errors)}")

    def _collect_investor(self, dates: list, market: str, sleep_sec: float):
        conn = sqlite3.connect(self.db_path)
        errors = []

        for i, day in enumerate(dates):
            day_str = day.replace("-", "")
            try:
                # Three separate calls: foreigner, institution, individual
                df_f = pykrx_stock.get_market_net_purchases_of_equities_by_ticker(
                    day_str, day_str, market=market, investor="외국인"
                )
                df_i = pykrx_stock.get_market_net_purchases_of_equities_by_ticker(
                    day_str, day_str, market=market, investor="기관합계"
                )
                df_p = pykrx_stock.get_market_net_purchases_of_equities_by_ticker(
                    day_str, day_str, market=market, investor="개인"
                )

                if df_f.empty and df_i.empty and df_p.empty:
                    self._log_collection(conn, day, "investor", 0)
                    continue

                # Merge on ticker index
                all_tickers = set(df_f.index) | set(df_i.index) | set(df_p.index)

                records = []
                for ticker in all_tickers:
                    foreigner = float(df_f.loc[ticker, "순매수거래량"]) if ticker in df_f.index else 0.0
                    institution = float(df_i.loc[ticker, "순매수거래량"]) if ticker in df_i.index else 0.0
                    individual = float(df_p.loc[ticker, "순매수거래량"]) if ticker in df_p.index else 0.0
                    records.append((day, ticker, foreigner, institution, individual))

                conn.executemany("""
                    INSERT OR REPLACE INTO daily_investor
                    (date, ticker, foreigner_net, institution_net, individual_net)
                    VALUES (?, ?, ?, ?, ?)
                """, records)

                self._log_collection(conn, day, "investor", len(records))

                if (i + 1) % 20 == 0 or i == 0:
                    remaining_est = (len(dates) - i - 1) * (2.5 + sleep_sec) / 60
                    print(f"  [{i+1}/{len(dates)}] {day} — {len(records)} tickers "
                          f"| ~{remaining_est:.1f} min left")

                time.sleep(sleep_sec)

            except Exception as e:
                errors.append((day, str(e)))
                print(f"  [{i+1}/{len(dates)}] {day} — ERROR: {e}")
                time.sleep(2)

        conn.close()
        if errors:
            print(f"  Investor errors: {len(errors)}")

    # ─────────────────────────────────────────────────────────
    # Querying
    # ─────────────────────────────────────────────────────────

    def get_screening_data(self, target_date: str) -> Dict:
        """
        Get everything the screener needs for a given date:
          - Last 60 days of OHLCV (for MAs)
          - Investor data for target_date + recent 5 days (for trend)

        Returns dict with DataFrames ready for screener.
        """
        conn = sqlite3.connect(self.db_path)

        # Last 61 trading days up to and including target_date
        dates_all = [r[0] for r in conn.execute("""
            SELECT DISTINCT date FROM daily_ohlcv
            WHERE date <= ?
            ORDER BY date DESC LIMIT 61
        """, [target_date]).fetchall()]

        if len(dates_all) < 6:
            conn.close()
            return {"error": f"Only {len(dates_all)} trading days available"}

        oldest = dates_all[-1]

        ohlcv = pd.read_sql_query("""
            SELECT * FROM daily_ohlcv
            WHERE date >= ? AND date <= ?
            ORDER BY ticker, date
        """, conn, params=[oldest, target_date])

        investor_today = pd.read_sql_query(
            "SELECT * FROM daily_investor WHERE date = ?",
            conn, params=[target_date]
        )

        recent_5 = dates_all[:5]
        if recent_5:
            investor_recent = pd.read_sql_query("""
                SELECT * FROM daily_investor
                WHERE date >= ? AND date <= ?
                ORDER BY ticker, date
            """, conn, params=[recent_5[-1], recent_5[0]])
        else:
            investor_recent = pd.DataFrame()

        conn.close()

        return {
            "target_date": target_date,
            "ohlcv": ohlcv,
            "investor_today": investor_today,
            "investor_recent": investor_recent,
            "trading_dates": sorted(dates_all),
        }

    def get_ticker_history(
        self,
        ticker: str,
        n_days: int = 60,
        before_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get recent OHLCV for a single ticker, sorted ascending."""
        conn = sqlite3.connect(self.db_path)
        if before_date:
            df = pd.read_sql_query("""
                SELECT * FROM daily_ohlcv
                WHERE ticker = ? AND date < ?
                ORDER BY date DESC LIMIT ?
            """, conn, params=[ticker, before_date, n_days])
        else:
            df = pd.read_sql_query("""
                SELECT * FROM daily_ohlcv
                WHERE ticker = ?
                ORDER BY date DESC LIMIT ?
            """, conn, params=[ticker, n_days])
        conn.close()
        return df.sort_values("date").reset_index(drop=True)

    def get_investor_history(
        self,
        ticker: str,
        n_days: int = 10,
        before_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get recent investor breakdown for a single ticker."""
        conn = sqlite3.connect(self.db_path)
        if before_date:
            df = pd.read_sql_query("""
                SELECT * FROM daily_investor
                WHERE ticker = ? AND date < ?
                ORDER BY date DESC LIMIT ?
            """, conn, params=[ticker, before_date, n_days])
        else:
            df = pd.read_sql_query("""
                SELECT * FROM daily_investor
                WHERE ticker = ?
                ORDER BY date DESC LIMIT ?
            """, conn, params=[ticker, n_days])
        conn.close()
        return df.sort_values("date").reset_index(drop=True)

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    def _get_trading_days(self, start: str, end: str) -> list:
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start_dt, end_dt)]

    def _get_collected_dates(self, data_type: str) -> set:
        conn = sqlite3.connect(self.db_path)
        dates = {r[0] for r in conn.execute(
            "SELECT date FROM collection_log WHERE data_type = ?", [data_type]
        ).fetchall()}
        conn.close()
        return dates

    def _log_collection(self, conn, day: str, data_type: str, count: int):
        conn.execute("""
            INSERT OR REPLACE INTO collection_log (date, data_type, tickers_count, collected_at)
            VALUES (?, ?, ?, datetime('now'))
        """, (day, data_type, count))
        conn.commit()

    def get_date_range(self) -> tuple:
        conn = sqlite3.connect(self.db_path)
        result = conn.execute(
            "SELECT MIN(date), MAX(date) FROM collection_log WHERE data_type = 'ohlcv'"
        ).fetchone()
        conn.close()
        return result

    def stats(self):
        conn = sqlite3.connect(self.db_path)
        ohlcv_days = conn.execute(
            "SELECT COUNT(*) FROM collection_log WHERE data_type = 'ohlcv'"
        ).fetchone()[0]
        investor_days = conn.execute(
            "SELECT COUNT(*) FROM collection_log WHERE data_type = 'investor'"
        ).fetchone()[0]
        ohlcv_rows = conn.execute("SELECT COUNT(*) FROM daily_ohlcv").fetchone()[0]
        investor_rows = conn.execute("SELECT COUNT(*) FROM daily_investor").fetchone()[0]
        tickers = conn.execute(
            "SELECT COUNT(DISTINCT ticker) FROM daily_ohlcv"
        ).fetchone()[0]
        date_range = self.get_date_range()
        conn.close()

        print(f"Database: {self.db_path}")
        print(f"  OHLCV days: {ohlcv_days} | rows: {ohlcv_rows:,}")
        print(f"  Investor days: {investor_days} | rows: {investor_rows:,}")
        print(f"  Unique tickers: {tickers}")
        if date_range[0]:
            print(f"  Date range: {date_range[0]} to {date_range[1]}")