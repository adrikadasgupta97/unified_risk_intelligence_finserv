"""
Customer authentication using bcrypt password hashing and JWT tokens.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from jose import JWTError, jwt
from pydantic import BaseModel

from src.config import config
from src.data.database import _get_conn


class TokenData(BaseModel):
    customer_id: str
    name: str
    account_number: str
    customer_value_tier: str
    role: str = "customer"   # "customer" | "admin"


class AuthResult(BaseModel):
    success: bool
    token: Optional[str] = None
    customer: Optional[TokenData] = None
    message: str = ""


def register_customer(
    customer_id: str,
    name: str,
    email: str,
    account_number: str,
    password: str,
    customer_value_tier: str = "Standard",
    role: str = "customer",
) -> bool:
    from src.data.database import init_db
    init_db()

    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        with _get_conn() as conn:
            conn.execute(
                """INSERT INTO customers
                   (customer_id, name, email, account_number, password_hash, customer_value_tier, role, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (customer_id, name, email, account_number, password_hash,
                 customer_value_tier, role, datetime.utcnow().isoformat()),
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def authenticate_customer(email: str, password: str) -> AuthResult:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM customers WHERE email = ?", (email,)
        ).fetchone()

    if not row:
        return AuthResult(success=False, message="Invalid email or password.")

    row = dict(row)
    if not bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
        return AuthResult(success=False, message="Invalid email or password.")

    token_data = TokenData(
        customer_id=row["customer_id"],
        name=row["name"],
        account_number=row["account_number"],
        customer_value_tier=row["customer_value_tier"],
        role=row.get("role", "customer"),
    )
    token = _create_token(token_data)
    return AuthResult(success=True, token=token, customer=token_data, message="Login successful.")


def lookup_role_by_email(email: str) -> Optional[str]:
    """Returns role if email exists, else None."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT role FROM customers WHERE email = ?", (email,)
        ).fetchone()
    return dict(row)["role"] if row else None


def reset_password(email: str, new_password: str, account_number: str = "", name: str = "") -> AuthResult:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT customer_id, role, name, account_number FROM customers WHERE email = ?",
            (email,),
        ).fetchone()

    if not row:
        return AuthResult(success=False, message="No account found with that email.")

    row = dict(row)
    if row["role"] == "admin":
        if row["name"].strip().lower() != name.strip().lower():
            return AuthResult(success=False, message="Full name does not match our records.")
    else:
        if row["account_number"] != account_number:
            return AuthResult(success=False, message="Account number does not match our records.")

    new_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    with _get_conn() as conn:
        conn.execute(
            "UPDATE customers SET password_hash = ? WHERE email = ?",
            (new_hash, email),
        )
        conn.commit()
    return AuthResult(success=True, message="Password reset successfully. Please login with your new password.")


def change_password(customer_id: str, current_password: str, new_password: str) -> AuthResult:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT password_hash FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()

    if not row:
        return AuthResult(success=False, message="Account not found.")

    if not bcrypt.checkpw(current_password.encode(), dict(row)["password_hash"].encode()):
        return AuthResult(success=False, message="Current password is incorrect.")

    new_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    with _get_conn() as conn:
        conn.execute(
            "UPDATE customers SET password_hash = ? WHERE customer_id = ?",
            (new_hash, customer_id),
        )
        conn.commit()
    return AuthResult(success=True, message="Password changed successfully.")


def verify_token(token: str) -> Optional[TokenData]:
    try:
        payload = jwt.decode(token, config.JWT_SECRET_KEY, algorithms=[config.security.jwt_algorithm])
        return TokenData(**payload)
    except JWTError:
        return None


def _create_token(data: TokenData) -> str:
    expire = datetime.utcnow() + timedelta(minutes=config.security.access_token_expire_minutes)
    payload = {**data.dict(), "exp": expire}
    return jwt.encode(payload, config.JWT_SECRET_KEY, algorithm=config.security.jwt_algorithm)
