"""
일일티커 v2 — Probability Model

Logistic regression models that predict the probability of a ticker
hitting +1.5%, +2%, +2.5%, +3% from next-day open.

Features:
  - foreigner_trend (0-5 days positive out of last 5)
  - institution_buying (binary)
  - volume_ratio (today's volume / 20-day avg)
  - breakout_distance (% from yesterday's high)
  - foreigner_cumulative (sum of last 5 days net buy, normalized)
  - change_pct (today's price change %)

Each threshold gets its own logistic regression model.

Usage:
    from scanner.probability_model import ProbabilityModel

    model = ProbabilityModel()
    model.train(data, train_dates)
    probabilities = model.predict(features)
    calibration = model.calibration_report(data, test_dates)
"""

import numpy as np
import pandas as pd
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import calibration_curve
except ImportError:
    raise ImportError("Install scikit-learn: pip install scikit-learn --break-system-packages")


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

THRESHOLDS = [1.5, 2.0, 2.5, 3.0]

# Screener gate (same as before — must pass to even be evaluated)
MIN_PRICE = 1_000
MIN_AVG_VOLUME = 50_000

# Feature names (order matters — must match build_features)
FEATURE_NAMES = [
    # Original
    "foreigner_trend",
    "institution_buying",
    "volume_ratio",
    "breakout_distance",
    "foreigner_cumulative_norm",
    "change_pct",
    # Momentum / Price Action
    "consecutive_up_days",
    "range_compression",
    "gap_from_ma20_pct",
    "upper_shadow_ratio",
    "return_5d",
    # Volume
    "volume_trend_3d",
    "relative_volume_spike",
    # Investor intensity
    "foreigner_intensity",
    "institution_intensity",
    "individual_net_negative",
    "smart_money_alignment",
]


class ProbabilityModel:
    def __init__(self):
        self.models = {}       # threshold -> LogisticRegression
        self.scalers = {}      # threshold -> StandardScaler
        self.trained = False

    def train(self, all_data: dict, train_dates: list, gate_params: dict = None):
        """
        Train one logistic regression per threshold.

        Args:
            all_data: preloaded data dict (same format as walk_forward)
            train_dates: list of date strings to train on
            gate_params: screener gate params (optional, uses defaults)
        """
        if gate_params is None:
            gate_params = {
                "volume_min": 1.0,
                "breakout_min": -3.0,
                "breakout_max": 20.0,
                "foreigner_trend_min": 2,
                "require_institution": False,
            }

        print(f"  Building training dataset from {len(train_dates)} dates...")

        # Build feature matrix and labels
        X_rows = []
        Y_rows = {t: [] for t in THRESHOLDS}

        eval_dates = train_dates[60:]  # Need 60 days for MAs
        n_samples = 0
        n_dates_with_data = 0

        for di, target_date in enumerate(eval_dates):
            candidates = _screen_and_extract(target_date, all_data, gate_params)

            if not candidates:
                continue

            n_dates_with_data += 1

            for cand in candidates:
                features = cand["features"]
                labels = cand["labels"]

                if labels is None:
                    continue

                X_rows.append(features)
                for t in THRESHOLDS:
                    Y_rows[t].append(labels[t])
                n_samples += 1

            if (di + 1) % 100 == 0:
                print(f"    [{di+1}/{len(eval_dates)}] {n_samples} samples so far")

        if n_samples < 50:
            print(f"  ERROR: Only {n_samples} samples, need at least 50.")
            return

        X = np.array(X_rows)
        print(f"  Training samples: {n_samples} from {n_dates_with_data} dates")
        print(f"  Feature matrix shape: {X.shape}")

        # Train one model per threshold
        for threshold in THRESHOLDS:
            Y = np.array(Y_rows[threshold])
            pos_rate = Y.mean() * 100

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train logistic regression
            model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )
            model.fit(X_scaled, Y)

            self.models[threshold] = model
            self.scalers[threshold] = scaler

            # Training accuracy
            train_probs = model.predict_proba(X_scaled)[:, 1]
            train_pred = (train_probs >= 0.5).astype(int)
            accuracy = (train_pred == Y).mean() * 100

            print(f"    +{threshold}%: base_rate={pos_rate:.1f}% accuracy={accuracy:.1f}% "
                  f"coefs={dict(zip(FEATURE_NAMES, model.coef_[0].round(3)))}")

        self.trained = True
        print(f"  Training complete.")

    def predict(self, features: np.ndarray) -> Dict[float, float]:
        """
        Predict probabilities for all thresholds.

        Args:
            features: 1D array of shape (n_features,) or 2D (n_samples, n_features)

        Returns:
            Dict mapping threshold -> probability (0-1)
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() first.")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        result = {}
        for threshold in THRESHOLDS:
            X_scaled = self.scalers[threshold].transform(features)
            probs = self.models[threshold].predict_proba(X_scaled)[:, 1]
            result[threshold] = probs if len(probs) > 1 else probs[0]

        return result

    def predict_single(self, feature_dict: dict) -> Dict[float, float]:
        """
        Predict from a feature dictionary.

        Args:
            feature_dict: dict with keys matching FEATURE_NAMES

        Returns:
            Dict mapping threshold -> probability (0-100%)
        """
        features = np.array([feature_dict[f] for f in FEATURE_NAMES])
        raw = self.predict(features)
        return {t: round(p * 100, 1) for t, p in raw.items()}

    def calibration_report(
        self,
        all_data: dict,
        test_dates: list,
        gate_params: dict = None,
        n_bins: int = 5,
    ) -> Dict:
        """
        Test calibration on out-of-sample dates.
        Groups predictions into bins and checks actual hit rates.
        """
        if gate_params is None:
            gate_params = {
                "volume_min": 1.0,
                "breakout_min": -3.0,
                "breakout_max": 20.0,
                "foreigner_trend_min": 2,
                "require_institution": False,
            }

        print(f"  Calibration test on {len(test_dates)} dates...")

        X_rows = []
        Y_rows = {t: [] for t in THRESHOLDS}

        eval_dates = [d for d in test_dates if d > test_dates[0]]  # Skip first few

        for target_date in eval_dates:
            candidates = _screen_and_extract(target_date, all_data, gate_params)
            for cand in candidates:
                if cand["labels"] is None:
                    continue
                X_rows.append(cand["features"])
                for t in THRESHOLDS:
                    Y_rows[t].append(cand["labels"][t])

        if len(X_rows) < 20:
            print(f"  Not enough test samples ({len(X_rows)})")
            return {}

        X = np.array(X_rows)
        n = len(X_rows)
        print(f"  Test samples: {n}")

        report = {}

        for threshold in THRESHOLDS:
            Y = np.array(Y_rows[threshold])
            X_scaled = self.scalers[threshold].transform(X)
            probs = self.models[threshold].predict_proba(X_scaled)[:, 1]

            actual_rate = Y.mean() * 100

            # Brier score
            brier = np.mean((probs - Y) ** 2)

            # Bin calibration
            bins = []
            bin_edges = np.linspace(0, 1, n_bins + 1)
            for i in range(n_bins):
                lo, hi = bin_edges[i], bin_edges[i + 1]
                mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
                if mask.sum() == 0:
                    continue
                predicted_mean = probs[mask].mean() * 100
                actual_mean = Y[mask].mean() * 100
                count = mask.sum()
                bins.append({
                    "range": f"{lo*100:.0f}-{hi*100:.0f}%",
                    "predicted": round(predicted_mean, 1),
                    "actual": round(actual_mean, 1),
                    "gap": round(actual_mean - predicted_mean, 1),
                    "count": int(count),
                })

            report[threshold] = {
                "brier": round(brier, 4),
                "actual_rate": round(actual_rate, 1),
                "avg_predicted": round(probs.mean() * 100, 1),
                "bins": bins,
            }

            print(f"    +{threshold}%: actual={actual_rate:.1f}% predicted_avg={probs.mean()*100:.1f}% brier={brier:.4f}")
            for b in bins:
                print(f"      {b['range']:>10}: predicted={b['predicted']:.1f}% actual={b['actual']:.1f}% "
                      f"gap={b['gap']:+.1f}% (n={b['count']})")

        return report

    def save(self, path: str):
        """Save trained models to disk."""
        with open(path, "wb") as f:
            pickle.dump({
                "models": self.models,
                "scalers": self.scalers,
                "thresholds": THRESHOLDS,
                "feature_names": FEATURE_NAMES,
            }, f)
        print(f"  Model saved to {path}")

    def load(self, path: str):
        """Load trained models from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.models = data["models"]
        self.scalers = data["scalers"]
        self.trained = True
        print(f"  Model loaded from {path}")


# ─────────────────────────────────────────────────────────────
# Feature Extraction + Labeling
# ─────────────────────────────────────────────────────────────

def _screen_and_extract(
    target_date: str,
    data: dict,
    gate_params: dict,
) -> List[Dict]:
    """
    Screen one day and extract features + labels for each passing ticker.
    Labels are based on NEXT day's open-to-high.
    """
    all_dates = data["all_dates"]
    date_to_idx = data["date_to_idx"]
    investor_today = data["investor_by_date"].get(target_date)

    target_idx = date_to_idx.get(target_date)
    if target_idx is None or target_idx < 61:
        return []

    trend_dates = all_dates[target_idx - 5:target_idx]

    results = []

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
        if vol_ratio < gate_params["volume_min"]:
            continue

        # Breakout
        yesterday_high = td["high"][pos - 1]
        if yesterday_high <= 0:
            continue
        brk_dist = (close - yesterday_high) / yesterday_high * 100
        if brk_dist < gate_params["breakout_min"] or brk_dist > gate_params["breakout_max"]:
            continue

        # Investor
        if investor_today is not None and ticker in investor_today.index:
            fgn = investor_today.loc[ticker, "foreigner_net"]
            inst = investor_today.loc[ticker, "institution_net"]
        else:
            fgn = 0
            inst = 0

        if fgn <= 0:
            continue

        if gate_params.get("require_institution", False) and inst <= 0:
            continue

        # Foreigner trend
        fgn_positive = 0
        fgn_cumulative = 0.0
        for td_date in trend_dates:
            inv_day = data["investor_by_date"].get(td_date)
            if inv_day is not None and ticker in inv_day.index:
                day_fgn = inv_day.loc[ticker, "foreigner_net"]
                if day_fgn > 0:
                    fgn_positive += 1
                fgn_cumulative += day_fgn

        if fgn_positive < gate_params.get("foreigner_trend_min", 2):
            continue

        # Normalize foreigner cumulative by avg volume
        fgn_cum_norm = fgn_cumulative / avg_vol if avg_vol > 0 else 0

        # Change pct
        prev_close = td["close"][pos - 1] if pos > 0 else 0
        if prev_close > 0:
            change_pct = (close - prev_close) / prev_close * 100
        else:
            change_pct = 0

        # ── New features: Momentum / Price Action ──

        # Consecutive up days (close > prev close streak)
        consecutive_up = 0
        for k in range(pos, max(pos - 10, 0), -1):
            if k > 0 and td["close"][k] > td["close"][k - 1]:
                consecutive_up += 1
            else:
                break

        # Range compression: today's range vs 20-day avg range
        today_range = (td["high"][pos] - td["low"][pos]) / close * 100 if close > 0 else 0
        ranges_20 = []
        for k in range(pos - 20, pos):
            if k >= 0 and td["close"][k] > 0:
                r = (td["high"][k] - td["low"][k]) / td["close"][k] * 100
                ranges_20.append(r)
        avg_range_20 = np.mean(ranges_20) if ranges_20 else 1
        range_compression = today_range / avg_range_20 if avg_range_20 > 0 else 1

        # Gap from MA20 (%)
        gap_from_ma20 = (close - ma20) / ma20 * 100 if ma20 > 0 else 0

        # Upper shadow ratio: (high - close) / (high - low)
        day_span = td["high"][pos] - td["low"][pos]
        if day_span > 0:
            upper_shadow = (td["high"][pos] - close) / day_span
        else:
            upper_shadow = 0

        # 5-day return
        if pos >= 5 and td["close"][pos - 5] > 0:
            return_5d = (close - td["close"][pos - 5]) / td["close"][pos - 5] * 100
        else:
            return_5d = 0

        # ── New features: Volume ──

        # Volume trend 3d: avg volume last 3 days / avg volume days 4-6
        if pos >= 6:
            vol_recent = np.mean(td["volume"][pos - 3:pos])
            vol_prior = np.mean(td["volume"][pos - 6:pos - 3])
            volume_trend_3d = vol_recent / vol_prior if vol_prior > 0 else 1
        else:
            volume_trend_3d = 1

        # Relative volume spike: today / max(last 20 days)
        max_vol_20 = np.max(td["volume"][pos - 20:pos]) if pos >= 20 else volume
        relative_vol_spike = volume / max_vol_20 if max_vol_20 > 0 else 0

        # ── New features: Investor intensity ──

        # Foreigner intensity: net / volume today
        foreigner_intensity = fgn / volume if volume > 0 else 0

        # Institution intensity: net / volume today
        institution_intensity = inst / volume if volume > 0 else 0

        # Individual net negative (binary: retail selling while smart money buys)
        if investor_today is not None and ticker in investor_today.index:
            indiv_net = investor_today.loc[ticker, "individual_net"]
        else:
            indiv_net = 0
        individual_net_negative = 1.0 if indiv_net < 0 else 0.0

        # Smart money alignment: both foreigner AND institution buying
        smart_money_alignment = 1.0 if (fgn > 0 and inst > 0) else 0.0

        # Features (17 total)
        features = [
            # Original
            fgn_positive,               # foreigner_trend (0-5)
            1.0 if inst > 0 else 0.0,   # institution_buying (binary)
            vol_ratio,                   # volume_ratio
            brk_dist,                    # breakout_distance
            fgn_cum_norm,                # foreigner_cumulative_norm
            change_pct,                  # change_pct
            # Momentum / Price Action
            consecutive_up,              # consecutive_up_days
            range_compression,           # range_compression
            gap_from_ma20,               # gap_from_ma20_pct
            upper_shadow,                # upper_shadow_ratio
            return_5d,                   # return_5d
            # Volume
            volume_trend_3d,             # volume_trend_3d
            relative_vol_spike,          # relative_volume_spike
            # Investor intensity
            foreigner_intensity,         # foreigner_intensity
            institution_intensity,       # institution_intensity
            individual_net_negative,     # individual_net_negative
            smart_money_alignment,       # smart_money_alignment
        ]

        # Labels: did next day hit each threshold from open?
        labels = _compute_labels(td, pos)

        results.append({
            "ticker": ticker,
            "pos": pos,
            "features": features,
            "labels": labels,
            "feature_dict": dict(zip(FEATURE_NAMES, features)),
        })

    return results


def _compute_labels(td: dict, pos: int) -> dict:
    """
    Check if next day's high reached each threshold from next day's open.
    Returns dict: threshold -> 0 or 1, or None if no next day data.
    """
    next_pos = pos + 1
    if next_pos >= len(td["dates"]):
        return None

    next_open = td["open"][next_pos]
    next_high = td["high"][next_pos]

    if next_open <= 0:
        return None

    max_gain_pct = (next_high - next_open) / next_open * 100

    labels = {}
    for threshold in THRESHOLDS:
        labels[threshold] = 1 if max_gain_pct >= threshold else 0

    return labels


# ─────────────────────────────────────────────────────────────
# Preloader (reuses walk_forward format)
# ─────────────────────────────────────────────────────────────

def preload_data(db_path: str) -> dict:
    """Load all data into memory."""
    print("  Loading data...")
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

    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    print(f"  {len(ticker_data)} tickers, {len(all_dates)} days loaded.")

    return {
        "all_dates": all_dates,
        "investor_by_date": investor_by_date,
        "ticker_data": ticker_data,
        "date_to_idx": date_to_idx,
    }


# ─────────────────────────────────────────────────────────────
# Walk-Forward Calibration Test
# ─────────────────────────────────────────────────────────────

def walk_forward_calibration(db_path: str = "data/daily_cache.db"):
    """
    Run walk-forward calibration across 5 folds.
    Train on 2020-X, test calibration on X+1.
    """
    folds = [
        {"train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31", "name": "2022"},
        {"train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31", "name": "2023"},
        {"train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31", "name": "2024"},
        {"train_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2025-12-31", "name": "2025"},
        {"train_end": "2025-12-31", "test_start": "2026-01-01", "test_end": "2026-12-31", "name": "2026"},
    ]

    data = preload_data(db_path)
    all_dates = data["all_dates"]

    all_fold_results = []

    for fold in folds:
        print(f"\n{'─' * 60}")
        print(f"  Fold: Train ≤{fold['train_end']} | Test {fold['name']}")
        print(f"{'─' * 60}")

        train_dates = [d for d in all_dates if d <= fold["train_end"]]
        test_dates = [d for d in all_dates if fold["test_start"] <= d <= fold["test_end"]]

        if len(test_dates) < 10:
            print(f"  SKIP: only {len(test_dates)} test dates")
            continue

        model = ProbabilityModel()
        model.train(data, train_dates)

        if not model.trained:
            print(f"  SKIP: training failed")
            continue

        report = model.calibration_report(data, test_dates)
        all_fold_results.append({
            "fold": fold["name"],
            "report": report,
        })

    # Summary
    print(f"\n\n{'=' * 70}")
    print(f"  CALIBRATION SUMMARY ACROSS ALL FOLDS")
    print(f"{'=' * 70}")

    for threshold in THRESHOLDS:
        print(f"\n  +{threshold}% threshold:")
        print(f"  {'Fold':<6} {'Actual':>8} {'Predicted':>10} {'Brier':>8}")
        print(f"  {'─' * 35}")

        briers = []
        for fr in all_fold_results:
            r = fr["report"].get(threshold, {})
            if not r:
                continue
            print(f"  {fr['fold']:<6} {r['actual_rate']:>7.1f}% {r['avg_predicted']:>9.1f}% {r['brier']:>8.4f}")
            briers.append(r["brier"])

        if briers:
            print(f"  {'AVG':<6} {'':>8} {'':>10} {np.mean(briers):>8.4f}")

    return all_fold_results


if __name__ == "__main__":
    results = walk_forward_calibration()