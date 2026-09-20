"""
Unit tests for the NLP pipeline components.
Run: pytest tests/
"""
from src.nlp.intent_detector import detect_intent
from src.nlp.entity_extractor import extract_entities
from src.security.pii_handler import detect_and_anonymize


def test_intent_register_complaint():
    result = detect_intent("I want to file a complaint about an unauthorized charge on my card.")
    assert result.intent == "register_complaint"
    assert result.confidence > 0.6


def test_intent_track_complaint():
    result = detect_intent("What is the status of my complaint CMP-ABC12345?")
    assert result.intent == "track_complaint"


def test_intent_escalate():
    result = detect_intent("I want to speak with a senior manager about this issue.")
    assert result.intent == "escalate_to_agent"


def test_entity_extract_amount():
    entities = extract_entities("I was charged ₹5000 on 12 Jan 2025 for a transaction I did not make.")
    assert entities.amount is not None
    assert "5000" in entities.amount


def test_entity_extract_complaint_id():
    entities = extract_entities("Please check complaint CMP-ABCD1234 for me.")
    assert entities.complaint_id_ref == "CMP-ABCD1234"


def test_pii_anonymization_phone():
    result = detect_and_anonymize("My phone number is 9876543210 and email is test@example.com")
    assert "9876543210" not in result.anonymized_text
    assert "test@example.com" not in result.anonymized_text


def test_pii_anonymization_card():
    result = detect_and_anonymize("My card number is 4111 1111 1111 1111")
    assert "4111" not in result.anonymized_text
