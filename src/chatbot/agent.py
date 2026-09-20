"""
Main complaint management agent.
Orchestrates: NLP → Classification → Risk Scoring → RAG → Escalation Check → DB persistence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.data.database import (
    get_complaint,
    get_customer_complaints,
    init_db,
    save_complaint,
    update_complaint,
)
from src.data.models import (
    ComplaintRecord,
    ComplaintStatus,
    ComplaintUpdate,
)
from src.nlp.nlp_pipeline import run_nlp_pipeline
from src.classification.complaint_classifier import classify_complaint
from src.classification.risk_scorer import compute_risk_score
from src.security.pii_handler import detect_and_anonymize
from src.rag.retriever import retrieve_relevant_articles
from src.rag.response_generator import generate_mitigation_response
from src.chatbot.escalation_engine import check_escalation


@dataclass
class AgentResponse:
    message: str
    complaint_id: Optional[str] = None
    escalated: bool = False
    risk_score: Optional[float] = None
    priority: Optional[str] = None
    category: Optional[str] = None
    secondary_docs: list = None   # list of {title, content} for expandable KB refs
    metrics: dict = None          # full risk/nlp breakdown for UI display


class ComplaintAgent:
    def __init__(self):
        init_db()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def register_complaint(
        self,
        customer_id: str,
        complaint_text: str,
        channel: str = "chatbot",
        customer_name: str = "valued customer",
        customer_value_tier: str = "Standard",
        unresolved_count: int = 0,
        complaint_frequency_30d: int = 1,
    ) -> AgentResponse:
        # 1. PII anonymization
        pii_result = detect_and_anonymize(complaint_text)

        # 2. NLP pipeline
        nlp_out = run_nlp_pipeline(pii_result.anonymized_text)

        # 3. Classification
        cls_result = classify_complaint(pii_result.anonymized_text)

        # 4. Risk scoring
        risk_result = compute_risk_score(
            category=cls_result.category.value,
            complaint_text=pii_result.anonymized_text,
            sentiment_polarity=nlp_out.sentiment.polarity,
            escalation_count=0,
            unresolved_count=unresolved_count,
            complaint_frequency_30d=complaint_frequency_30d,
            customer_value_tier=customer_value_tier,
        )

        # 5. RAG — retrieve KB articles
        retrieved = retrieve_relevant_articles(
            query=pii_result.anonymized_text,
            category_hint=cls_result.category.value,
        )

        # 6. Escalation check
        escalation = check_escalation(
            risk_score=risk_result.total_score,
            category=cls_result.category.value,
            resolution_attempts=0,
            escalation_count=0,
            customer_requested=False,
        )

        # 7. Generate mitigation response (only if not immediately escalating)
        if escalation.should_escalate:
            mitigation = (
                f"I understand this is an urgent matter, {customer_name}. "
                f"Given the nature and priority of your complaint, I am immediately escalating this to "
                f"a senior support agent. {escalation.reason} "
                f"You will be contacted within 2 hours. Your complaint has been registered."
            )
            mitigation_db = (
                "The customer's complaint has been escalated to a senior support agent. "
                f"{escalation.reason} "
                "The customer will be contacted within 2 hours."
            )
        else:
            mitigation = generate_mitigation_response(
                complaint_text=pii_result.anonymized_text,
                retrieved_docs=retrieved,
                category=cls_result.category.value,
                customer_name=customer_name,
            )
            # Strip customer name from DB-stored version to avoid PII in admin views
            mitigation_db = mitigation.replace(customer_name, "the customer") if customer_name else mitigation

        # 8. Persist to DB
        complaint = ComplaintRecord(
            customer_id=customer_id,
            raw_text=complaint_text,
            anonymized_text=pii_result.anonymized_text,
            channel=channel,
            **nlp_out.to_complaint_fields(),
            category=cls_result.category,
            priority=risk_result.priority,
            risk_score=risk_result.total_score,
            status=ComplaintStatus.ESCALATED if escalation.should_escalate else ComplaintStatus.IN_PROGRESS,
            escalated=escalation.should_escalate,
            escalation_count=1 if escalation.should_escalate else 0,
            resolution_attempts=0 if escalation.should_escalate else 1,
            mitigation_response=mitigation_db,
        )
        complaint_id = save_complaint(complaint)

        secondary = [
            {"title": doc.title, "content": doc.content}
            for doc in retrieved[1:]
        ] if not escalation.should_escalate else []

        metrics = {
            "category": cls_result.category.value,
            "category_confidence": round(cls_result.confidence, 3),
            "intent": nlp_out.intent.intent,
            "sentiment": nlp_out.sentiment.label.value,
            "sentiment_polarity": round(nlp_out.sentiment.polarity, 3),
            "priority": risk_result.priority.value,
            **{k: round(v, 3) for k, v in risk_result.breakdown().items()
               if k not in ("priority",)},
            "escalated": escalation.should_escalate,
            "escalation_reason": escalation.reason or "None",
            "kb_articles_retrieved": len(retrieved),
        }

        return AgentResponse(
            message=mitigation,
            complaint_id=complaint_id,
            escalated=escalation.should_escalate,
            risk_score=risk_result.total_score,
            priority=risk_result.priority.value,
            category=cls_result.category.value,
            secondary_docs=secondary,
            metrics=metrics,
        )

    def track_complaint(self, complaint_id: str, customer_id: str) -> AgentResponse:
        complaint = get_complaint(complaint_id)

        if not complaint:
            return AgentResponse(
                message=f"No complaint found with ID **{complaint_id}**. "
                        "Please verify the reference number and try again."
            )

        if complaint.customer_id != customer_id:
            return AgentResponse(
                message="You are not authorized to view this complaint. "
                        "Please provide the complaint ID registered under your account."
            )

        status_msg = (
            f"Here is the status for complaint **{complaint_id}**:\n\n"
            f"- **Status:** {complaint.status}\n"
            f"- **Category:** {complaint.category or 'Being assessed'}\n"
            f"- **Priority:** {complaint.priority or 'Being assessed'}\n"
            f"- **Registered on:** {complaint.created_at.strftime('%d %b %Y, %I:%M %p')} UTC\n"
            f"- **Last Updated:** {complaint.updated_at.strftime('%d %b %Y, %I:%M %p')} UTC\n"
        )

        if complaint.status == ComplaintStatus.RESOLVED:
            status_msg += f"\nYour complaint has been resolved. We hope this has addressed your concern."
        elif complaint.escalated:
            status_msg += f"\nYour complaint has been escalated to a senior agent and is being actively worked on."
        elif complaint.mitigation_response:
            status_msg += f"\n**Suggested Resolution:**\n{complaint.mitigation_response}"

        return AgentResponse(
            message=status_msg,
            complaint_id=complaint_id,
            escalated=complaint.escalated,
            risk_score=complaint.risk_score,
            priority=complaint.priority,
            category=complaint.category,
        )

    def list_customer_complaints(self, customer_id: str) -> AgentResponse:
        complaints = get_customer_complaints(customer_id)
        if not complaints:
            return AgentResponse(message="You have no registered complaints.")

        lines = ["Here are your complaints:\n"]
        for c in complaints:
            lines.append(
                f"- **{c.complaint_id}** | {c.status} | {c.category or 'Uncategorized'} "
                f"| Priority: {c.priority or 'TBD'} | {c.created_at.strftime('%d %b %Y')}"
            )
        return AgentResponse(message="\n".join(lines))

    def escalate_complaint(self, complaint_id: str, customer_id: str) -> AgentResponse:
        complaint = get_complaint(complaint_id)
        if not complaint or complaint.customer_id != customer_id:
            return AgentResponse(message="Complaint not found or access denied.")

        updated = update_complaint(
            complaint_id,
            ComplaintUpdate(
                status=ComplaintStatus.ESCALATED,
                escalated=True,
            ),
        )
        if updated:
            updated.escalation_count += 1
            save_complaint(updated)

        return AgentResponse(
            message=f"Complaint **{complaint_id}** has been escalated to a senior support agent. "
                    "You will be contacted within 2 hours.",
            complaint_id=complaint_id,
            escalated=True,
        )

    def resolve_complaint(self, complaint_id: str) -> AgentResponse:
        update_complaint(complaint_id, ComplaintUpdate(status=ComplaintStatus.RESOLVED))
        return AgentResponse(
            message=f"Complaint **{complaint_id}** has been marked as resolved. Thank you for your feedback.",
            complaint_id=complaint_id,
        )
