"""
FastAPI application — REST API for the complaint management agent.
"""
from __future__ import annotations

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from src.chatbot.agent import ComplaintAgent
from src.security.auth import authenticate_customer, register_customer, verify_token, TokenData
from src.data.database import get_customer_complaints, init_db
from src.rag.kb_ingestion import ingest_knowledge_base

app = FastAPI(
    title="Intelligent Complaint Management Agent",
    description="AI-powered complaint registration, tracking, classification and mitigation for financial services.",
    version="1.0.0",
)

_bearer = HTTPBearer()
_agent = ComplaintAgent()


@app.on_event("startup")
def startup():
    init_db()
    try:
        n = ingest_knowledge_base()
        if n:
            print(f"[Startup] Ingested {n} KB articles into ChromaDB.")
    except Exception as e:
        print(f"[Startup] KB ingestion skipped: {e}")


def _get_current_customer(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> TokenData:
    customer = verify_token(credentials.credentials)
    if not customer:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token.")
    return customer


# ------------------------------------------------------------------ #
# Auth endpoints                                                       #
# ------------------------------------------------------------------ #

class RegisterRequest(BaseModel):
    customer_id: str
    name: str
    email: str
    account_number: str
    password: str
    customer_value_tier: str = "Standard"


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/auth/register", tags=["Auth"])
def register(req: RegisterRequest):
    ok = register_customer(
        req.customer_id, req.name, req.email,
        req.account_number, req.password, req.customer_value_tier,
    )
    if not ok:
        raise HTTPException(status_code=400, detail="Email or account number already registered.")
    return {"message": "Registration successful."}


@app.post("/auth/login", tags=["Auth"])
def login(req: LoginRequest):
    result = authenticate_customer(req.email, req.password)
    if not result.success:
        raise HTTPException(status_code=401, detail=result.message)
    return {"token": result.token, "customer": result.customer.dict()}


# ------------------------------------------------------------------ #
# Complaint endpoints                                                  #
# ------------------------------------------------------------------ #

class ComplaintRequest(BaseModel):
    complaint_text: str
    channel: str = "api"


@app.post("/complaints/register", tags=["Complaints"])
def register_complaint(
    req: ComplaintRequest,
    customer: TokenData = Depends(_get_current_customer),
):
    existing = get_customer_complaints(customer.customer_id)
    unresolved = sum(1 for c in existing if c.status not in ("Resolved", "Closed"))

    response = _agent.register_complaint(
        customer_id=customer.customer_id,
        complaint_text=req.complaint_text,
        channel=req.channel,
        customer_name=customer.name,
        customer_value_tier=customer.customer_value_tier,
        unresolved_count=unresolved,
        complaint_frequency_30d=len(existing),
    )
    return {
        "complaint_id": response.complaint_id,
        "category": response.category,
        "priority": response.priority,
        "risk_score": response.risk_score,
        "escalated": response.escalated,
        "message": response.message,
    }


@app.get("/complaints/{complaint_id}", tags=["Complaints"])
def track_complaint(
    complaint_id: str,
    customer: TokenData = Depends(_get_current_customer),
):
    response = _agent.track_complaint(complaint_id, customer.customer_id)
    return {"message": response.message, "escalated": response.escalated}


@app.get("/complaints", tags=["Complaints"])
def list_complaints(customer: TokenData = Depends(_get_current_customer)):
    complaints = get_customer_complaints(customer.customer_id)
    return {"complaints": [c.dict() for c in complaints]}


@app.post("/complaints/{complaint_id}/escalate", tags=["Complaints"])
def escalate(
    complaint_id: str,
    customer: TokenData = Depends(_get_current_customer),
):
    response = _agent.escalate_complaint(complaint_id, customer.customer_id)
    return {"message": response.message}


@app.post("/complaints/{complaint_id}/resolve", tags=["Complaints"])
def resolve(
    complaint_id: str,
    customer: TokenData = Depends(_get_current_customer),
):
    response = _agent.resolve_complaint(complaint_id)
    return {"message": response.message}


# ------------------------------------------------------------------ #
# Health                                                               #
# ------------------------------------------------------------------ #

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "service": "Complaint Management Agent"}
