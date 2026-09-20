"""
Submit high-severity complaints for Arjun Singh and Rahul Sharma
to push their churn scores into the High risk tier (>=65%).
"""
import uuid
import sys
sys.path.insert(0, ".")

from src.security.auth import TokenData
from src.chatbot.conversation_manager import ConversationManager

PROMPTS = {
    "Arjun Singh": {
        "customer_id": "CUST-DE50BC",
        "tier": "Platinum",
        "prompts": [
            "My account has been completely compromised. Someone has transferred Rs.75,000 out of my account without my knowledge or consent and I am unable to log in.",
            "I have raised this fraud complaint three times and no one has taken any action. This is absolutely unacceptable and I am extremely frustrated.",
            "My entire savings of Rs.1,20,000 have been wiped out due to unauthorized transactions. I demand immediate escalation to the highest authority.",
            "Despite multiple escalations my money has not been returned. I am filing a police complaint and will approach the banking ombudsman if this is not resolved today.",
        ],
    },
    "Rahul Sharma": {
        "customer_id": "CUST-BCFDCA",
        "tier": "Gold",
        "prompts": [
            "There have been 5 unauthorized international transactions on my card totaling Rs.95,000. I never travel abroad and these are clearly fraudulent.",
            "I reported the fraud 48 hours ago and my card is still active. No one has blocked it or refunded the money. This is a complete failure of your security.",
            "I am being charged interest on amounts I never spent due to the fraudulent transactions. Your bank is making me pay for your security failure.",
            "I have been calling every day for a week with no resolution. The agent today was rude and disconnected the call. I am extremely angry and will close all my accounts.",
        ],
    },
}

total = 0
for name, data in PROMPTS.items():
    customer = TokenData(
        customer_id=data["customer_id"],
        name=name,
        email="",
        account_number="",
        customer_value_tier=data["tier"],
        role="customer",
    )
    session_id = str(uuid.uuid4())
    cm = ConversationManager()
    cm.start_session(customer, session_id)

    print(f"\n{'='*60}")
    print(f"Customer: {name} ({data['customer_id']})")
    print(f"{'='*60}")

    for i, prompt in enumerate(data["prompts"], 1):
        response = cm.handle_message(session_id, prompt)
        cid   = response.complaint_id or "—"
        cat   = response.category or "—"
        pri   = response.priority or "—"
        risk  = f"{response.risk_score*100:.1f}%" if response.risk_score else "—"
        esc   = "YES" if response.escalated else "no"
        print(f"  [{i}] {cid} | {cat} | {pri} | Risk: {risk} | Escalated: {esc}")
        total += 1

print(f"\nDone. {total} complaints submitted (left Unresolved intentionally).")
