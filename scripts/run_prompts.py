"""
Batch-submit complaint prompts for multiple customers through the agent pipeline.
"""
import uuid
import sys
sys.path.insert(0, ".")

from src.security.auth import TokenData
from src.chatbot.conversation_manager import ConversationManager

PROMPTS = {
    "Amit Kumar": {
        "customer_id": "CUST-E8482A",
        "tier": "Gold",
        "prompts": [
            "Someone used my card details online and made purchases worth Rs.15,000 without my knowledge.",
            "My EMI bounce charge of Rs.500 was applied even though my account had sufficient balance on the due date.",
            "The interest rate on my personal loan was changed without any prior communication from the bank.",
            "A payment of Rs.12,000 was deducted from my account but the merchant says they never received it.",
            "I have been on hold for over 45 minutes and my issue has still not been addressed.",
        ],
    },
    "Neha Gupta": {
        "customer_id": "CUST-931D62",
        "tier": "Gold",
        "prompts": [
            "I have been locked out of my account after entering the wrong password three times.",
            "I changed my phone number and can no longer log in because OTPs go to my old number.",
            "I made a purchase of Rs.10,000 last week but the reward points have not been credited to my account.",
            "My reward points expired without any prior notification or reminder from your side.",
            "I was charged a late payment fee of Rs.500 even though I paid on time. Please waive it.",
        ],
    },
    "Arjun Singh": {
        "customer_id": "CUST-DE50BC",
        "tier": "Platinum",
        "prompts": [
            "I want to foreclose my personal loan but the agent told me there is a penalty which was not mentioned in my loan agreement.",
            "My loan account is showing an incorrect outstanding balance that does not match my repayment schedule.",
            "I received an OTP I did not request and now there is an unauthorized charge of Rs.7,500 on my account.",
            "My credit limit was reduced from Rs.1,00,000 to Rs.50,000 without any prior notice.",
            "My statement shows a wrong outstanding balance. The amount is much higher than what I actually owe.",
        ],
    },
    "Shreya Das": {
        "customer_id": "CUST-8E79A1",
        "tier": "Platinum",
        "prompts": [
            "I applied for a replacement card 15 days ago but have not received it yet.",
            "My card expired last month and I still haven't received my renewed card.",
            "I tried to redeem 5,000 reward points for a voucher but the portal shows an error every time.",
            "What is the minimum amount due on my credit card this month?",
        ],
    },
    "Sandeep Chatterjee": {
        "customer_id": "CUST-451CC3",
        "tier": "Platinum",
        "prompts": [
            "I made a transaction of Rs.8,000 on 10 June but it still hasn't reflected in my statement.",
            "My UPI transfer of Rs.5,000 to my friend failed but the money was deducted.",
            "I am not receiving the OTP on my registered mobile number to complete login.",
            "I applied for a pre-approved loan 10 days ago but have not received any update on the status.",
            "I was promised 10x reward points on a specific merchant but only received 1x points.",
        ],
    },
    "Vikram Patel": {
        "customer_id": "CUST-40A6DB",
        "tier": "Standard",
        "prompts": [
            "How do I activate my new credit card after receiving it?",
            "What are the charges for cash withdrawal using my credit card?",
            "I need to increase my credit limit as I have a large purchase coming up next week.",
            "My cashback of Rs.800 from last month's offer has not been credited to my account.",
        ],
    },
    "Anjali Roy": {
        "customer_id": "CUST-35F0FF",
        "tier": "Standard",
        "prompts": [
            "My card has been blocked and I don't know why. I need it unblocked urgently.",
            "My account shows as suspended when I try to access the app. I have not violated any terms.",
            "I never shopped at this merchant but there is a charge of Rs.3,200 from them on my statement.",
            "I got a refund of Rs.2,000 from Amazon but it hasn't been credited to my account yet.",
        ],
    },
    "Aditi Nair": {
        "customer_id": "CUST-650CF6",
        "tier": "Standard",
        "prompts": [
            "How many reward points do I need to redeem for a Rs.500 voucher?",
            "What is the grace period for paying my credit card bill without interest?",
            "My new card is not working for contactless payments even though it should support NFC.",
        ],
    },
}

total_submitted = 0
total_failed = 0

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
    print(f"Customer: {name} ({data['customer_id']}) — {len(data['prompts'])} prompts")
    print(f"{'='*60}")

    for i, prompt in enumerate(data["prompts"], 1):
        try:
            response = cm.handle_message(session_id, prompt)
            cid = response.complaint_id or "—"
            cat = response.category or "—"
            pri = response.priority or "—"
            risk = f"{response.risk_score*100:.1f}%" if response.risk_score else "—"
            print(f"  [{i}] {cid} | {cat} | {pri} | Risk: {risk}")
            print(f"       Prompt: {prompt[:70]}...")
            total_submitted += 1
        except Exception as e:
            print(f"  [{i}] ERROR: {e}")
            total_failed += 1

print(f"\n{'='*60}")
print(f"Done. Submitted: {total_submitted}  Failed: {total_failed}")
print(f"{'='*60}")
