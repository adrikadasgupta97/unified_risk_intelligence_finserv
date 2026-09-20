"""
Customer Churn Analyzer.

Computes a churn probability score per customer from their complaint interaction history.

Features used:
  1.  complaint_frequency_30d    — complaints in last 30 days  (volume signal)
  2.  complaint_frequency_90d    — complaints in last 90 days  (sustained volume)
  3.  avg_sentiment_polarity     — mean polarity across all complaints (-1 to +1)
  4.  sentiment_trend            — slope of polarity over time (negative = worsening)
  5.  escalation_rate            — escalated / total complaints
  6.  unresolved_rate            — unresolved / total complaints
  7.  avg_risk_score             — mean complaint risk score
  8.  max_risk_score             — worst single complaint risk score
  9.  critical_count             — number of Critical-priority complaints
  10. days_since_last_complaint  — recency (lower = more engaged / frustrated)

Churn score ∈ [0.0, 1.0].  Tiers: High ≥ 0.65 | Medium ≥ 0.50 | Low < 0.50
Thresholds are calibrated to the empirical score distribution of this dataset.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.data.database import DB_PATH

_CHURN_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "churn_classifier.pkl"

_churn_model_cache = None
_churn_model_loaded = False


def _get_trained_churn_model():
    global _churn_model_cache, _churn_model_loaded
    if not _churn_model_loaded:
        _churn_model_loaded = True
        if _CHURN_MODEL_PATH.exists():
            import joblib
            _churn_model_cache = joblib.load(_CHURN_MODEL_PATH)
    return _churn_model_cache


# ------------------------------------------------------------------ #
# Feature weights (sum to 1.0)                                        #
# ------------------------------------------------------------------ #
_WEIGHTS = {
    "avg_sentiment_polarity": 0.20,   # most direct dissatisfaction signal
    "sentiment_trend":        0.15,   # worsening trend matters more than snapshot
    "escalation_rate":        0.15,   # escalations show broken trust
    "unresolved_rate":        0.15,   # unresolved issues drive churn
    "avg_risk_score":         0.12,   # risk severity
    "max_risk_score":         0.08,   # extreme complaint experienced
    "complaint_frequency_30d":0.08,   # recent surge
    "complaint_frequency_90d":0.04,   # longer window
    "critical_count":         0.02,   # critical complaints
    "days_since_last_complaint": 0.01,# recency (small weight — ambiguous signal)
}

# Normalisation caps for each feature
_CAPS = {
    "complaint_frequency_30d":  10,
    "complaint_frequency_90d":  20,
    "critical_count":           5,
    "days_since_last_complaint": 90,  # beyond 90 days → treated as inactive, low risk
}

_CHURN_HIGH   = 0.65
_CHURN_MEDIUM = 0.50


@dataclass
class CustomerChurnFeatures:
    customer_id: str
    name: str
    total_complaints: int
    complaint_frequency_30d: int
    complaint_frequency_90d: int
    avg_sentiment_polarity: float
    sentiment_trend: float          # negative = getting worse
    escalation_rate: float
    unresolved_rate: float
    avg_risk_score: float
    max_risk_score: float
    critical_count: int
    days_since_last_complaint: int
    churn_score: float = 0.0
    churn_risk: str = "Low"

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class ChurnReport:
    customers: list[CustomerChurnFeatures] = field(default_factory=list)
    high_risk_count: int = 0
    medium_risk_count: int = 0
    low_risk_count: int = 0
    avg_churn_score: float = 0.0


def compute_churn_report() -> ChurnReport:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Load all complaints with customer names
    rows = conn.execute("""
        SELECT c.*, cu.name
        FROM complaints c
        LEFT JOIN customers cu ON c.customer_id = cu.customer_id
        ORDER BY c.customer_id, c.created_at ASC
    """).fetchall()
    conn.close()

    if not rows:
        return ChurnReport()

    # Group by customer
    from collections import defaultdict
    customer_complaints: dict[str, list[dict]] = defaultdict(list)
    customer_names: dict[str, str] = {}
    for r in rows:
        d = dict(r)
        customer_complaints[d["customer_id"]].append(d)
        customer_names[d["customer_id"]] = d.get("name") or d["customer_id"]

    report = ChurnReport()
    now = datetime.utcnow()
    cutoff_30d = now - timedelta(days=30)
    cutoff_90d = now - timedelta(days=90)

    for cid, complaints in customer_complaints.items():
        features = _extract_features(
            cid, customer_names[cid], complaints, now, cutoff_30d, cutoff_90d
        )
        features.churn_score = _score(features)
        features.churn_risk  = _predict_tier(features)
        report.customers.append(features)

    report.customers.sort(key=lambda x: x.churn_score, reverse=True)
    report.high_risk_count   = sum(1 for c in report.customers if c.churn_risk == "High")
    report.medium_risk_count = sum(1 for c in report.customers if c.churn_risk == "Medium")
    report.low_risk_count    = sum(1 for c in report.customers if c.churn_risk == "Low")
    report.avg_churn_score   = round(
        float(np.mean([c.churn_score for c in report.customers])), 4
    ) if report.customers else 0.0

    return report


def _extract_features(
    cid: str,
    name: str,
    complaints: list[dict],
    now: datetime,
    cutoff_30d: datetime,
    cutoff_90d: datetime,
) -> CustomerChurnFeatures:
    total = len(complaints)

    freq_30d = sum(
        1 for c in complaints
        if _parse_dt(c["created_at"]) >= cutoff_30d
    )
    freq_90d = sum(
        1 for c in complaints
        if _parse_dt(c["created_at"]) >= cutoff_90d
    )

    polarities = [
        float(c["sentiment_score"])
        for c in complaints
        if c["sentiment_score"] is not None
    ]
    avg_polarity = float(np.mean(polarities)) if polarities else 0.0
    sentiment_trend = _compute_trend(polarities)

    escalation_rate = sum(1 for c in complaints if c["escalated"]) / total
    unresolved_rate = sum(
        1 for c in complaints if c["status"] not in ("Resolved", "Closed")
    ) / total

    risk_scores = [
        float(c["risk_score"])
        for c in complaints
        if c["risk_score"] is not None
    ]
    avg_risk = float(np.mean(risk_scores)) if risk_scores else 0.0
    max_risk = float(max(risk_scores)) if risk_scores else 0.0

    critical_count = sum(1 for c in complaints if c.get("priority") == "Critical")

    last_dt = _parse_dt(complaints[-1]["created_at"])
    days_since = (now - last_dt).days

    return CustomerChurnFeatures(
        customer_id=cid,
        name=name,
        total_complaints=total,
        complaint_frequency_30d=freq_30d,
        complaint_frequency_90d=freq_90d,
        avg_sentiment_polarity=round(avg_polarity, 4),
        sentiment_trend=round(sentiment_trend, 4),
        escalation_rate=round(escalation_rate, 4),
        unresolved_rate=round(unresolved_rate, 4),
        avg_risk_score=round(avg_risk, 4),
        max_risk_score=round(max_risk, 4),
        critical_count=critical_count,
        days_since_last_complaint=days_since,
    )


def _score(f: CustomerChurnFeatures) -> float:
    """Map each feature to [0,1] churn contribution, then take weighted sum."""
    def norm(val, cap):
        return min(val / cap, 1.0)

    components = {
        # Sentiment polarity: -1 → 1.0 churn, +1 → 0.0 churn
        "avg_sentiment_polarity":  (1.0 - f.avg_sentiment_polarity) / 2.0,
        # Trend: slope in [-1,1]; map negative trend → higher churn
        "sentiment_trend":         (1.0 - max(min(f.sentiment_trend, 1.0), -1.0)) / 2.0,
        "escalation_rate":         f.escalation_rate,
        "unresolved_rate":         f.unresolved_rate,
        "avg_risk_score":          f.avg_risk_score,
        "max_risk_score":          f.max_risk_score,
        "complaint_frequency_30d": norm(f.complaint_frequency_30d, _CAPS["complaint_frequency_30d"]),
        "complaint_frequency_90d": norm(f.complaint_frequency_90d, _CAPS["complaint_frequency_90d"]),
        "critical_count":          norm(f.critical_count, _CAPS["critical_count"]),
        # Recent complaints = more frustrated; no complaints recently = may have left already
        "days_since_last_complaint": 1.0 - norm(f.days_since_last_complaint, _CAPS["days_since_last_complaint"]),
    }

    score = sum(_WEIGHTS[k] * v for k, v in components.items())
    return round(min(score, 1.0), 4)


def _predict_tier(f: "CustomerChurnFeatures") -> str:
    """Return tier from trained RF if model is available, else fall back to rule-based thresholds."""
    bundle = _get_trained_churn_model()
    if bundle is not None:
        try:
            row = np.array([[
                f.complaint_frequency_30d, f.complaint_frequency_90d,
                f.avg_sentiment_polarity,  f.sentiment_trend,
                f.escalation_rate,         f.unresolved_rate,
                f.avg_risk_score,          f.max_risk_score,
                f.critical_count,          f.days_since_last_complaint,
            ]])
            return str(bundle["model"].predict(row)[0])
        except Exception:
            pass
    return _tier(f.churn_score)


def _tier(score: float) -> str:
    if score >= _CHURN_HIGH:
        return "High"
    elif score >= _CHURN_MEDIUM:
        return "Medium"
    return "Low"


def _compute_trend(values: list[float]) -> float:
    """Linear regression slope over a sequence of polarity values."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    slope = float(np.polyfit(x, values, 1)[0])
    return round(slope, 4)


def _parse_dt(dt_str: str) -> datetime:
    try:
        return datetime.fromisoformat(dt_str)
    except Exception:
        return datetime.utcnow()
