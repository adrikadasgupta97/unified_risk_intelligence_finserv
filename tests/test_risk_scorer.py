from src.classification.risk_scorer import compute_risk_score
from src.data.models import Priority


def test_fraud_complaint_scores_high():
    result = compute_risk_score(
        category="Fraud & Unauthorized Charges",
        complaint_text="There is an unauthorized transaction on my account. This is urgent!",
        sentiment_polarity=-0.9,
        escalation_count=1,
        unresolved_count=2,
        complaint_frequency_30d=3,
        customer_value_tier="Platinum",
    )
    assert result.total_score >= 0.75
    assert result.priority in (Priority.CRITICAL, Priority.HIGH)


def test_general_query_scores_low():
    result = compute_risk_score(
        category="General Inquiry",
        complaint_text="I have a question about my rewards points.",
        sentiment_polarity=0.2,
        escalation_count=0,
        unresolved_count=0,
        complaint_frequency_30d=1,
        customer_value_tier="Standard",
    )
    assert result.total_score < 0.50


def test_score_bounded_0_to_1():
    result = compute_risk_score(
        category="Fraud & Unauthorized Charges",
        complaint_text="URGENT! Legal action! Emergency! Elderly customer! RBI complaint!",
        sentiment_polarity=-1.0,
        escalation_count=10,
        unresolved_count=10,
        complaint_frequency_30d=20,
        customer_value_tier="Platinum",
    )
    assert 0.0 <= result.total_score <= 1.0
