import json
import sqlite3
from datetime import datetime
from typing import Optional

from src.config import config
from src.data.models import ComplaintRecord, ComplaintUpdate, ComplaintSummary, ComplaintStatus


DB_PATH = config.DATA_DIR / "complaints.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS complaints (
                complaint_id       TEXT PRIMARY KEY,
                customer_id        TEXT NOT NULL,
                raw_text           TEXT NOT NULL,
                anonymized_text    TEXT,
                channel            TEXT DEFAULT 'chatbot',
                intent             TEXT,
                entities           TEXT,
                sentiment          TEXT,
                sentiment_score    REAL,
                category           TEXT,
                priority           TEXT,
                risk_score         REAL,
                status             TEXT DEFAULT 'Open',
                escalated          INTEGER DEFAULT 0,
                escalation_count   INTEGER DEFAULT 0,
                resolution_attempts INTEGER DEFAULT 0,
                assigned_agent     TEXT,
                mitigation_response TEXT,
                created_at         TEXT NOT NULL,
                updated_at         TEXT NOT NULL,
                resolved_at        TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id        TEXT PRIMARY KEY,
                name               TEXT NOT NULL,
                email              TEXT UNIQUE NOT NULL,
                account_number     TEXT UNIQUE NOT NULL,
                password_hash      TEXT NOT NULL,
                customer_value_tier TEXT DEFAULT 'Standard',
                role               TEXT DEFAULT 'customer',
                created_at         TEXT NOT NULL
            )
        """)
        # Migrate existing DBs that don't have the role column yet
        try:
            conn.execute("ALTER TABLE customers ADD COLUMN role TEXT DEFAULT 'customer'")
            conn.commit()
        except Exception:
            pass
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_complaints_customer
            ON complaints(customer_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_complaints_status
            ON complaints(status)
        """)
        conn.commit()


def save_complaint(complaint: ComplaintRecord) -> str:
    with _get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO complaints VALUES (
                :complaint_id, :customer_id, :raw_text, :anonymized_text, :channel,
                :intent, :entities, :sentiment, :sentiment_score,
                :category, :priority, :risk_score,
                :status, :escalated, :escalation_count, :resolution_attempts,
                :assigned_agent, :mitigation_response,
                :created_at, :updated_at, :resolved_at
            )
        """, {
            **complaint.dict(),
            "entities": json.dumps(complaint.entities) if complaint.entities else None,
            "escalated": int(complaint.escalated),
            "created_at": complaint.created_at.isoformat(),
            "updated_at": complaint.updated_at.isoformat(),
            "resolved_at": complaint.resolved_at.isoformat() if complaint.resolved_at else None,
        })
        conn.commit()
    return complaint.complaint_id


def get_complaint(complaint_id: str) -> Optional[ComplaintRecord]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM complaints WHERE complaint_id = ?", (complaint_id,)
        ).fetchone()
    if not row:
        return None
    return _row_to_record(dict(row))


def get_customer_complaints(customer_id: str) -> list[ComplaintSummary]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM complaints WHERE customer_id = ? ORDER BY created_at DESC",
            (customer_id,)
        ).fetchall()
    return [
        ComplaintSummary(
            complaint_id=r["complaint_id"],
            status=r["status"],
            category=r["category"],
            priority=r["priority"],
            risk_score=r["risk_score"],
            created_at=datetime.fromisoformat(r["created_at"]),
            updated_at=datetime.fromisoformat(r["updated_at"]),
        )
        for r in rows
    ]


def update_complaint(complaint_id: str, update: ComplaintUpdate) -> Optional[ComplaintRecord]:
    fields = {k: v for k, v in update.dict(exclude_none=True).items()}
    if not fields:
        return get_complaint(complaint_id)

    fields["updated_at"] = datetime.utcnow().isoformat()
    if fields.get("status") == ComplaintStatus.RESOLVED:
        fields["resolved_at"] = datetime.utcnow().isoformat()
    if "escalated" in fields:
        fields["escalated"] = int(fields["escalated"])

    set_clause = ", ".join(f"{k} = :{k}" for k in fields)
    fields["complaint_id"] = complaint_id
    with _get_conn() as conn:
        conn.execute(
            f"UPDATE complaints SET {set_clause} WHERE complaint_id = :complaint_id",
            fields
        )
        conn.commit()
    return get_complaint(complaint_id)


def _row_to_record(row: dict) -> ComplaintRecord:
    row["entities"] = json.loads(row["entities"]) if row["entities"] else None
    row["escalated"] = bool(row["escalated"])
    row["created_at"] = datetime.fromisoformat(row["created_at"])
    row["updated_at"] = datetime.fromisoformat(row["updated_at"])
    row["resolved_at"] = datetime.fromisoformat(row["resolved_at"]) if row["resolved_at"] else None
    return ComplaintRecord(**row)
