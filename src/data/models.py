from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class ComplaintStatus(str, Enum):
    OPEN = "Open"
    IN_PROGRESS = "In Progress"
    ESCALATED = "Escalated"
    RESOLVED = "Resolved"
    CLOSED = "Closed"


class Priority(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class ComplaintCategory(str, Enum):
    TRANSACTION_DISPUTE = "Transaction Dispute"
    ACCOUNT_ACCESS = "Account Access Issue"
    BILLING = "Billing Concern"
    CARD_SERVICES = "Card Services"
    FRAUD = "Fraud & Unauthorized Charges"
    REWARDS = "Reward & Points Issue"
    CUSTOMER_SERVICE = "Customer Service Complaint"
    LOAN_CREDIT = "Loan & Credit Issue"
    GENERAL = "General Inquiry"


class ComplaintRecord(BaseModel):
    complaint_id: str = Field(default_factory=lambda: f"CMP-{uuid.uuid4().hex[:8].upper()}")
    customer_id: str
    raw_text: str
    anonymized_text: Optional[str] = None
    channel: str = "chatbot"

    # NLP outputs
    intent: Optional[str] = None
    entities: Optional[dict] = None
    sentiment: Optional[SentimentLabel] = None
    sentiment_score: Optional[float] = None

    # Classification outputs
    category: Optional[ComplaintCategory] = None
    priority: Optional[Priority] = None
    risk_score: Optional[float] = None

    # Lifecycle
    status: ComplaintStatus = ComplaintStatus.OPEN
    escalated: bool = False
    escalation_count: int = 0
    resolution_attempts: int = 0
    assigned_agent: Optional[str] = None
    mitigation_response: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None

    class Config:
        use_enum_values = True


class ComplaintUpdate(BaseModel):
    status: Optional[ComplaintStatus] = None
    priority: Optional[Priority] = None
    assigned_agent: Optional[str] = None
    mitigation_response: Optional[str] = None
    escalated: Optional[bool] = None
    resolution_attempts: Optional[int] = None


class ComplaintSummary(BaseModel):
    complaint_id: str
    status: str
    category: Optional[str]
    priority: Optional[str]
    risk_score: Optional[float]
    created_at: datetime
    updated_at: datetime
