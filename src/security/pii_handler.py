"""
PII detection and anonymization using Microsoft Presidio.
Masks entities like card numbers, phone numbers, email, names, SSN etc.
before storing complaint text in the database.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.config import config


@dataclass
class PIIResult:
    anonymized_text: str
    detected_entities: list[dict]   # [{entity_type, start, end, score}]


# Fallback regex patterns when Presidio is unavailable
_FALLBACK_PATTERNS = [
    (re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"), "[CARD_NUMBER]"),
    (re.compile(r"\b\d{9,18}\b"), "[ACCOUNT_NUMBER]"),
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE), "[EMAIL]"),
    (re.compile(r"\b(\+91|0)?[6-9]\d{9}\b"), "[PHONE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN_CARD]"),
    (re.compile(r"\b\d{12}\b"), "[AADHAAR]"),
]

_analyzer = None
_anonymizer = None


def _get_presidio():
    global _analyzer, _anonymizer
    if _analyzer is None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        _analyzer = AnalyzerEngine()
        _anonymizer = AnonymizerEngine()
    return _analyzer, _anonymizer


def detect_and_anonymize(text: str) -> PIIResult:
    try:
        return _presidio_anonymize(text)
    except Exception:
        return _regex_anonymize(text)


def _presidio_anonymize(text: str) -> PIIResult:
    analyzer, anonymizer = _get_presidio()
    entities = config.security.pii_entities

    results = analyzer.analyze(text=text, entities=entities, language="en")
    detected = [
        {"entity_type": r.entity_type, "start": r.start, "end": r.end, "score": r.score}
        for r in results
    ]

    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return PIIResult(anonymized_text=anonymized.text, detected_entities=detected)


def _regex_anonymize(text: str) -> PIIResult:
    anonymized = text
    detected = []
    for pattern, placeholder in _FALLBACK_PATTERNS:
        for match in pattern.finditer(anonymized):
            detected.append({
                "entity_type": placeholder.strip("[]"),
                "start": match.start(),
                "end": match.end(),
                "score": 0.85,
            })
        anonymized = pattern.sub(placeholder, anonymized)
    return PIIResult(anonymized_text=anonymized, detected_entities=detected)
