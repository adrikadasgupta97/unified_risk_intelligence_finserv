"""
Named entity extractor for financial complaint domain.
Extracts complaint-relevant entities: amounts, dates, transaction IDs, card types, etc.
Uses spaCy NER + regex for financial-domain entities.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExtractedEntities:
    complaint_id_ref: Optional[str] = None       # e.g., CMP-ABC12345
    amount: Optional[str] = None                 # e.g., ₹5000, $200
    transaction_id: Optional[str] = None
    date_mentioned: Optional[str] = None
    merchant: Optional[str] = None
    card_type: Optional[str] = None
    account_type: Optional[str] = None
    location: Optional[str] = None
    misc: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v}


_PATTERNS = {
    "complaint_id_ref": re.compile(r"\bCMP-[A-Z0-9]{6,10}\b"),
    "amount": re.compile(
        r"(?:₹|Rs\.?|INR|USD|\$)\s*[\d,]+(?:\.\d{1,2})?|\b[\d,]+(?:\.\d{1,2})?\s*(?:rupees?|dollars?)\b",
        re.IGNORECASE,
    ),
    "transaction_id": re.compile(
        r"\b(?:txn|transaction|ref|reference|utr|arn)[:\s#]*([A-Z0-9]{8,20})\b",
        re.IGNORECASE,
    ),
    "date_mentioned": re.compile(
        r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4}|yesterday|today|last\s+(?:week|month|monday|tuesday|wednesday|thursday|friday))\b",
        re.IGNORECASE,
    ),
    "card_type": re.compile(
        r"\b(credit card|debit card|prepaid card|amex card|platinum card|gold card|blue card|green card)\b",
        re.IGNORECASE,
    ),
    "account_type": re.compile(
        r"\b(savings account|current account|loan account|credit account|checking account)\b",
        re.IGNORECASE,
    ),
}

_MERCHANT_PATTERN = re.compile(
    r"(?:at|from|to|merchant[:\s]+|store[:\s]+)\s+([A-Z][A-Za-z0-9\s&']{2,30})",
    re.IGNORECASE,
)


def extract_entities(text: str) -> ExtractedEntities:
    entities = ExtractedEntities()

    for field_name, pattern in _PATTERNS.items():
        match = pattern.search(text)
        if match:
            # For transaction_id, use group 1 (the captured ID), else the full match
            value = match.group(1) if field_name == "transaction_id" and match.lastindex else match.group(0)
            setattr(entities, field_name, value.strip())

    merchant_match = _MERCHANT_PATTERN.search(text)
    if merchant_match:
        entities.merchant = merchant_match.group(1).strip()

    # Augment with spaCy if available (PERSON, GPE/LOC for location)
    try:
        _augment_with_spacy(text, entities)
    except Exception:
        pass

    return entities


def _augment_with_spacy(text: str, entities: ExtractedEntities):
    import spacy
    nlp = _get_spacy()
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC") and not entities.location:
            entities.location = ent.text
        elif ent.label_ == "MONEY" and not entities.amount:
            entities.amount = ent.text
        elif ent.label_ == "DATE" and not entities.date_mentioned:
            entities.date_mentioned = ent.text


_spacy_nlp = None


def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp
