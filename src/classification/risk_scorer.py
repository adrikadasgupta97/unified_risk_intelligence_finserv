"""
Complaint Risk Scorer.

Risk Score = weighted sum of:
  - severity_score     (derived from category + keywords)
  - sentiment_score    (from FinBERT polarity)
  - escalation_factor  (escalation_count / max_escalations)
  - unresolved_factor  (unresolved open complaints / threshold)
  - frequency_factor   (complaint count in last 30 days / threshold)
  - customer_value     (tier-based multiplier)

Final score ∈ [0.0, 1.0]. Higher = more critical.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.config import config
from src.data.models import ComplaintCategory, Priority

_RISK_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "risk_scorer.pkl"
_risk_model_cache = None
_risk_model_loaded = False


def _get_trained_risk_model():
    global _risk_model_cache, _risk_model_loaded
    if not _risk_model_loaded:
        _risk_model_loaded = True
        if _RISK_MODEL_PATH.exists():
            import joblib
            _risk_model_cache = joblib.load(_RISK_MODEL_PATH)
    return _risk_model_cache


_SEVERITY_BASE: dict[str, float] = {
    ComplaintCategory.FRAUD: 1.0,
    ComplaintCategory.TRANSACTION_DISPUTE: 0.80,
    ComplaintCategory.ACCOUNT_ACCESS: 0.70,
    ComplaintCategory.BILLING: 0.65,
    ComplaintCategory.LOAN_CREDIT: 0.65,
    ComplaintCategory.CARD_SERVICES: 0.55,
    ComplaintCategory.REWARDS: 0.45,
    ComplaintCategory.CUSTOMER_SERVICE: 0.50,
    ComplaintCategory.GENERAL: 0.30,
}

_SEVERITY_BOOSTERS = [
    (re.compile(r"\b(urgent|asap|immediately|emergency|critical)\b", re.IGNORECASE), 0.10),
    (re.compile(r"\b(large amount|huge|significant|substantial)\b", re.IGNORECASE), 0.05),
    (re.compile(r"\b(legal action|court|consumer forum|rbi|grievance)\b", re.IGNORECASE), 0.15),
    (re.compile(r"\b(elderly|senior citizen|medical|hospital)\b", re.IGNORECASE), 0.08),
]

_CUSTOMER_VALUE_MULTIPLIER: dict[str, float] = {
    "Platinum": 1.0,
    "Gold": 0.75,
    "Standard": 0.50,
}

_MAX_ESCALATIONS = 3
_MAX_UNRESOLVED = 5
_MAX_FREQUENCY = 10


@dataclass
class RiskScoreResult:
    total_score: float
    severity: float
    sentiment_component: float
    escalation_component: float
    unresolved_component: float
    frequency_component: float
    customer_value_component: float
    priority: Priority

    def breakdown(self) -> dict:
        return {
            "total_risk_score": round(self.total_score, 4),
            "severity": round(self.severity, 4),
            "sentiment": round(self.sentiment_component, 4),
            "escalation": round(self.escalation_component, 4),
            "unresolved": round(self.unresolved_component, 4),
            "frequency": round(self.frequency_component, 4),
            "customer_value": round(self.customer_value_component, 4),
            "priority": self.priority.value,
        }


def compute_risk_score(
    category: str,
    complaint_text: str,
    sentiment_polarity: float,    # [-1.0, +1.0]
    escalation_count: int = 0,
    unresolved_count: int = 0,
    complaint_frequency_30d: int = 1,
    customer_value_tier: str = "Standard",
    use_ml: bool = True,
) -> RiskScoreResult:
    weights = config.risk_scoring.weights

    # 1. Severity
    base_severity = _SEVERITY_BASE.get(category, 0.30)
    boost = sum(b for pattern, b in _SEVERITY_BOOSTERS if pattern.search(complaint_text))
    severity = min(base_severity + boost, 1.0)

    # 2. Sentiment — negative sentiment increases risk
    # polarity in [-1, 1]; map to [0, 1] where -1 → 1.0 risk, +1 → 0.0 risk
    sentiment_component = (1.0 - sentiment_polarity) / 2.0

    # 3. Escalation
    escalation_component = min(escalation_count / _MAX_ESCALATIONS, 1.0)

    # 4. Unresolved open complaints
    unresolved_component = min(unresolved_count / _MAX_UNRESOLVED, 1.0)

    # 5. Complaint frequency (last 30 days)
    frequency_component = min(complaint_frequency_30d / _MAX_FREQUENCY, 1.0)

    # 6. Customer value
    customer_value_component = _CUSTOMER_VALUE_MULTIPLIER.get(customer_value_tier, 0.50)

    total = (
        weights["severity"] * severity
        + weights["sentiment"] * sentiment_component
        + weights["escalation_frequency"] * escalation_component
        + weights["unresolved_cases"] * unresolved_component
        + weights["complaint_frequency"] * frequency_component
        + weights["customer_value"] * customer_value_component
    )
    total = round(min(total, 1.0), 4)

    ml_pred = _ml_priority(
        category, sentiment_component, escalation_component,
        unresolved_component, frequency_component, customer_value_component,
    ) if use_ml else None
    priority = ml_pred or _score_to_priority(total)

    return RiskScoreResult(
        total_score=total,
        severity=severity,
        sentiment_component=sentiment_component,
        escalation_component=escalation_component,
        unresolved_component=unresolved_component,
        frequency_component=frequency_component,
        customer_value_component=customer_value_component,
        priority=priority,
    )


def _ml_priority(
    category: str,
    sentiment_component: float,
    escalation_component: float,
    unresolved_component: float,
    frequency_component: float,
    customer_value_component: float,
) -> Priority | None:
    """Return ML-predicted priority when model is loaded, else None."""
    bundle = _get_trained_risk_model()
    if bundle is None:
        return None
    try:
        import numpy as np
        row = np.array([[
            _SEVERITY_BASE.get(category, 0.30),
            sentiment_component,
            escalation_component,
            unresolved_component,
            frequency_component,
            customer_value_component,
        ]])
        pred = str(bundle["model"].predict(row)[0])
        return Priority(pred)
    except Exception:
        return None


def _score_to_priority(score: float) -> Priority:
    thresholds = config.risk_scoring.thresholds
    if score >= thresholds["critical"]:
        return Priority.CRITICAL
    elif score >= thresholds["high"]:
        return Priority.HIGH
    elif score >= thresholds["medium"]:
        return Priority.MEDIUM
    return Priority.LOW
