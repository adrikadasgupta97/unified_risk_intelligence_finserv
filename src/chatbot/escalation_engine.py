"""
Determines whether a complaint should be escalated to a human agent.
Escalation is triggered by:
  - Risk score above threshold
  - High-risk category (e.g., Fraud)
  - Max auto-resolution attempts exceeded
  - Explicit customer request
  - Repeated escalation history
"""
from __future__ import annotations

from dataclasses import dataclass

from src.config import config


@dataclass
class EscalationDecision:
    should_escalate: bool
    reason: str


def check_escalation(
    risk_score: float,
    category: str,
    resolution_attempts: int,
    escalation_count: int,
    customer_requested: bool = False,
) -> EscalationDecision:
    if customer_requested:
        return EscalationDecision(
            should_escalate=True,
            reason="Customer requested to speak with a human agent.",
        )

    if category in config.escalation.escalation_categories:
        return EscalationDecision(
            should_escalate=True,
            reason=f"Complaint category '{category}' requires mandatory human review.",
        )

    if risk_score >= config.escalation.risk_score_threshold:
        return EscalationDecision(
            should_escalate=True,
            reason=f"High risk score ({risk_score:.2f}) exceeds escalation threshold.",
        )

    if resolution_attempts >= config.escalation.max_auto_resolution_attempts:
        return EscalationDecision(
            should_escalate=True,
            reason="Maximum automated resolution attempts reached without resolution.",
        )

    if escalation_count > 0:
        return EscalationDecision(
            should_escalate=True,
            reason="Complaint has been previously escalated and requires agent follow-up.",
        )

    return EscalationDecision(should_escalate=False, reason="")
