"""
Simulate ~200 complaints across 10 existing + 5 new customers.
Covers all complaint categories with varied severity and sentiment.
"""
import uuid
import sys
sys.path.insert(0, ".")

from src.security.auth import TokenData
from src.chatbot.conversation_manager import ConversationManager

# ------------------------------------------------------------------ #
# Customer roster                                                      #
# ------------------------------------------------------------------ #
CUSTOMERS = [
    # --- existing ---
    {"name": "Amit Kumar",         "customer_id": "CUST-E8482A",  "tier": "Gold"},
    {"name": "Neha Gupta",         "customer_id": "CUST-931D62",  "tier": "Gold"},
    {"name": "Arjun Singh",        "customer_id": "CUST-DE50BC",  "tier": "Platinum"},
    {"name": "Shreya Das",         "customer_id": "CUST-8E79A1",  "tier": "Platinum"},
    {"name": "Sandeep Chatterjee", "customer_id": "CUST-451CC3",  "tier": "Platinum"},
    {"name": "Vikram Patel",       "customer_id": "CUST-40A6DB",  "tier": "Standard"},
    {"name": "Anjali Roy",         "customer_id": "CUST-35F0FF",  "tier": "Standard"},
    {"name": "Aditi Nair",         "customer_id": "CUST-650CF6",  "tier": "Standard"},
    {"name": "Rahul Sharma",       "customer_id": "CUST-BCFDCA",  "tier": "Gold"},
    {"name": "Priya Menon",        "customer_id": "CUST-77957F",  "tier": "Gold"},
    # --- new ---
    {"name": "Karan Mehta",        "customer_id": "CUST-A1B2C3",  "tier": "Platinum"},
    {"name": "Divya Iyer",         "customer_id": "CUST-D4E5F6",  "tier": "Gold"},
    {"name": "Rohan Verma",        "customer_id": "CUST-G7H8I9",  "tier": "Standard"},
    {"name": "Sunita Reddy",       "customer_id": "CUST-J1K2L3",  "tier": "Platinum"},
    {"name": "Manish Joshi",       "customer_id": "CUST-M4N5O6",  "tier": "Gold"},
]

# ------------------------------------------------------------------ #
# Prompt bank — ~200 prompts spread across categories & severities    #
# ------------------------------------------------------------------ #
PROMPTS = {
    "Amit Kumar": [
        "I have been charged Rs.2,500 as a joining fee even though I was told this card has no joining fee.",
        "My auto-debit for credit card bill failed even though my account had sufficient balance.",
        "I received a message that my card was used for a transaction of Rs.45,000 which I did not make.",
        "The minimum amount due on my statement is showing incorrect figures this month.",
        "I applied for an instant loan but the amount was not disbursed even after 7 days.",
        "I am very satisfied with the resolution of my previous issue. The team was extremely helpful.",
    ],
    "Neha Gupta": [
        "My credit card was used for an international transaction of Rs.32,000 and I have never been abroad.",
        "I tried to block my card via the app after suspicious activity but the option is not working.",
        "My account shows a transaction reversal pending for 15 days but the amount has not been credited back.",
        "I was charged a foreign transaction fee on a purchase made from an Indian website.",
        "My card limit was supposed to be upgraded after 6 months but it has not been done.",
        "I am not able to generate my credit card PIN from the app or ATM.",
        "I have been waiting for my replacement card for 3 weeks now. This is unacceptable.",
    ],
    "Arjun Singh": [
        "There is a duplicate charge of Rs.8,000 appearing on my statement for a purchase I made only once.",
        "My loan EMI has been debited twice this month causing my account to go into negative balance.",
        "I have submitted my KYC documents three times and my account is still restricted.",
        "The interest charged on my credit card this month is much higher than the stated rate.",
        "I was promised a cashback of Rs.3,000 on my last purchase but it has not been credited.",
        "My complaint from last month has still not been resolved despite multiple follow-ups.",
    ],
    "Shreya Das": [
        "My credit score dropped by 80 points after your bank reported a late payment which I did not make.",
        "I paid my full outstanding balance but the bank still reported it as partially paid to CIBIL.",
        "I have set up autopay but my bill was not paid automatically this month and I was charged late fees.",
        "I cannot access my account online for the past 5 days. The portal keeps saying service unavailable.",
        "My add-on card holder made a transaction that I did not authorize. Please block the add-on card.",
        "The reward points I redeemed last week have been deducted but I have not received the voucher.",
        "Thank you for quickly resolving my account access issue. Really appreciate the support.",
    ],
    "Sandeep Chatterjee": [
        "I received a phishing call claiming to be from your bank asking for my OTP. I did not share it but want to report this.",
        "My card was cloned and used at multiple petrol pumps across different cities simultaneously.",
        "I have been receiving promotional SMS even after opting out multiple times.",
        "My statement is showing a merchant name I do not recognize for a charge of Rs.15,000.",
        "I was on hold for 2 hours and my call was disconnected without any resolution.",
        "The in-app chat has not responded to my message for 48 hours.",
        "My international transaction limit is blocked even though I informed the bank before travelling.",
    ],
    "Vikram Patel": [
        "How do I convert my outstanding balance into EMI?",
        "What is the process to add my spouse as an add-on cardholder?",
        "I want to know the current reward points balance on my account.",
        "My card was declined at a POS terminal even though I had sufficient limit.",
        "I accidentally paid Rs.10,000 extra on my bill. When will the excess amount be refunded?",
        "The ATM cash withdrawal limit on my card seems lower than what was promised.",
        "I am happy with the quick response on my refund request. Thank you!",
    ],
    "Anjali Roy": [
        "I received a call saying I won a prize and they asked for my card details. I hung up. Please flag this.",
        "My cashback offer from the shopping festival last month is still not credited.",
        "I tried to use my card at an international website and it was declined repeatedly.",
        "My card expired and the replacement has not arrived even after 30 days.",
        "A refund of Rs.5,500 from an online store has not been credited to my card for 20 days.",
        "I was charged a late payment fee even though I had set up standing instructions for minimum payment.",
        "I am unable to view my last 3 months of statements on the app.",
    ],
    "Aditi Nair": [
        "The mobile app keeps crashing when I try to view my transaction history.",
        "I am not receiving OTP on my registered mobile number. Please help.",
        "I want to close my credit card account. What is the process and are there any charges?",
        "My account has been flagged for suspicious activity and I cannot make any transactions.",
        "I made a payment to the wrong account by mistake. Can this be reversed?",
        "Thank you so much for resolving the OTP issue so quickly. The service has been excellent.",
        "I am very happy with the new app update. Everything is working smoothly now.",
    ],
    "Rahul Sharma": [
        "I have disputed a charge of Rs.28,000 over a month ago but there is no update on the status.",
        "The bank deducted Rs.1,200 as annual fee even though I had crossed the spending threshold for fee waiver.",
        "My pre-approved personal loan offer has expired without any prior notice.",
        "I was charged for travel insurance on my card which I never opted for.",
        "I have been asking for a No-Objection Certificate for my closed loan for 6 weeks with no response.",
        "My card rewards tier was downgraded without any communication from the bank.",
        "Three of my complaints are still unresolved. I am extremely disappointed with the service.",
    ],
    "Priya Menon": [
        "I received my credit card statement with an incorrect billing cycle date.",
        "My home loan EMI increased without any prior notice regarding interest rate change.",
        "I was incorrectly charged a cheque bounce fee even though I never issued a cheque.",
        "My fixed deposit matured but the amount has not been credited to my linked account.",
        "The bank branch told me I needed to visit in person for a simple address change which is very inconvenient.",
        "I have not received my income tax certificate for last year despite multiple requests.",
    ],
    # --- new customers ---
    "Karan Mehta": [
        "I lost my wallet and need to block my credit card immediately. Please help.",
        "Someone has changed my registered email without my knowledge. I am locked out of all services.",
        "Rs.75,000 was transferred out of my savings account overnight. I did not initiate this.",
        "My Platinum card benefits are not being applied at the airport lounge. I was turned away.",
        "I have been incorrectly classified as a defaulter in your system. My credit score is being impacted.",
        "The bank charged me a late fee of Rs.1,000 even though the payment was deducted via standing instruction on the due date.",
        "My card was used for 6 transactions in 10 minutes across different cities — clearly fraudulent.",
        "I escalated this fraud issue 5 days ago and nobody has called me back. This is a serious security breach.",
    ],
    "Divya Iyer": [
        "I applied for a credit limit increase 3 weeks ago. There is no update and no communication.",
        "My cashback of Rs.2,000 from the festive offer expired without being credited.",
        "I cannot redeem my reward points — the app says technical error every time.",
        "My credit card bill payment went through but the outstanding balance is still showing.",
        "I received a fraud alert SMS for a transaction I did make. Please do not block my card.",
        "The interest free period on my card was incorrectly calculated this month.",
        "I am satisfied with the resolution. The agent was very professional and helpful.",
        "My complaint was resolved faster than expected. Great service this time around.",
    ],
    "Rohan Verma": [
        "I have been charged twice for the same UPI payment of Rs.3,000.",
        "My salary account was debited for an EMI I had already foreclosed.",
        "I want to know the status of my home loan application submitted 2 weeks ago.",
        "The bank's app does not support UPI payments above Rs.25,000. This limit is too low.",
        "My credit card statement shows a purchase I made abroad in the wrong currency conversion rate.",
        "I was not informed about the change in minimum balance requirement for my savings account.",
        "I received a cheque book I never requested. Please ensure no unauthorized service requests are made.",
    ],
    "Sunita Reddy": [
        "My Platinum travel card is not accepted at certain merchants even though it is a Visa card.",
        "I have a dispute on a hotel booking charge of Rs.18,000 where the hotel cancelled my reservation.",
        "My loan prepayment was not applied to the principal — the statement still shows the full amount.",
        "I made a balance transfer from another bank but the amount is not showing in my account.",
        "I was told my credit card annual fee would be waived if I spend Rs.2 lakh per year. I did but the fee was still charged.",
        "My account statement shows transactions from a location I have never visited.",
        "The bank executive who called me was very rude when I asked about my complaint status.",
        "I have submitted the same document 4 times for my home loan and each time I am told it is insufficient.",
        "My insurance premium linked to this card was debited twice in the same month.",
    ],
    "Manish Joshi": [
        "I cannot complete my KYC update online even though I have all the required documents.",
        "My card was hot-listed by the bank without informing me and I was embarrassed at a merchant.",
        "I received an SMS about a loan disbursement I never applied for.",
        "My reward points are expiring next week but I cannot find any redemption option that works.",
        "I was charged a processing fee for a balance transfer that was advertised as fee-free.",
        "The bank blocked my international transactions without prior notice before my overseas trip.",
        "My credit utilisation report to CIBIL is incorrect — it shows a higher outstanding than actual.",
        "I have been trying to reach the grievance officer for 10 days with no success.",
        "Thank you for finally resolving the international transaction block before my trip. Very helpful.",
    ],
}

# ------------------------------------------------------------------ #
# Run                                                                  #
# ------------------------------------------------------------------ #
customer_map = {c["name"]: c for c in CUSTOMERS}
total_ok = 0
total_err = 0

for name, prompts in PROMPTS.items():
    cdata = customer_map[name]
    customer = TokenData(
        customer_id=cdata["customer_id"],
        name=name,
        email="",
        account_number="",
        customer_value_tier=cdata["tier"],
        role="customer",
    )
    print(f"\n{'='*60}")
    print(f"{name} ({cdata['customer_id']}) — {cdata['tier']} — {len(prompts)} prompts")
    print(f"{'='*60}")

    for i, prompt in enumerate(prompts, 1):
        try:
            session_id = str(uuid.uuid4())
            cm = ConversationManager()
            cm.start_session(customer, session_id)
            r = cm.handle_message(session_id, prompt)
            cid  = r.complaint_id or "—"
            cat  = r.category or "—"
            pri  = r.priority or "—"
            risk = f"{r.risk_score*100:.1f}%" if r.risk_score else "—"
            print(f"  [{i:02d}] {cid} | {cat} | {pri} | {risk}")
            total_ok += 1
        except Exception as e:
            print(f"  [{i:02d}] ERROR: {e}")
            total_err += 1

print(f"\n{'='*60}")
print(f"Done. Submitted: {total_ok}  Errors: {total_err}")
print(f"{'='*60}")
