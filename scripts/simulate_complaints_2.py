"""
Simulate ~340 additional complaints across the 15 existing customers.
Balanced across all 9 complaint categories.
"""
import uuid
import sys
sys.path.insert(0, ".")

from src.security.auth import TokenData
from src.chatbot.conversation_manager import ConversationManager

CUSTOMERS = [
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
    {"name": "Karan Mehta",        "customer_id": "CUST-A1B2C3",  "tier": "Platinum"},
    {"name": "Divya Iyer",         "customer_id": "CUST-D4E5F6",  "tier": "Gold"},
    {"name": "Rohan Verma",        "customer_id": "CUST-G7H8I9",  "tier": "Standard"},
    {"name": "Sunita Reddy",       "customer_id": "CUST-J1K2L3",  "tier": "Platinum"},
    {"name": "Manish Joshi",       "customer_id": "CUST-M4N5O6",  "tier": "Gold"},
]

PROMPTS = {
    # ~23 prompts each across 15 customers ≈ 345 total

    "Amit Kumar": [
        # Fraud & Unauthorized
        "Someone has made an international purchase of Rs.12,000 on my card without my knowledge.",
        "I received an OTP for a transaction I did not initiate. The money has already been deducted.",
        # Billing
        "My credit card statement shows a finance charge that should have been waived per my plan.",
        "I was charged a late payment fee even though I paid before the due date and have the screenshot.",
        "The minimum amount due shown on my statement contradicts what the app shows.",
        # Transaction Dispute
        "I paid Rs.7,500 at a restaurant but the charge on my card shows Rs.9,000.",
        "A merchant refund of Rs.4,200 was supposed to hit my card 10 days ago and has not arrived.",
        "I disputed a transaction 3 weeks ago and have not received any update on the chargeback.",
        # Card Services
        "My new credit card arrived but the chip is not working at any POS terminal.",
        "I want to upgrade my card to the Platinum variant but the app does not show the option.",
        "My card was declined at a grocery store even though I have Rs.50,000 limit available.",
        # Account Access
        "I am getting an error 'Account Suspended' when trying to log in to the mobile app.",
        "Two-factor authentication is not working. The OTP is delivered but the app rejects it.",
        # Loan & Credit
        "My personal loan pre-closure request was submitted but the outstanding amount has not updated.",
        "The credit limit on my card was reduced from Rs.80,000 to Rs.40,000 without any notice.",
        # Reward & Points
        "I redeemed 10,000 points for a flight voucher but received a hotel voucher instead.",
        "My reward points from the last 3 months are not showing in the app.",
        # Customer Service
        "The agent I spoke to was extremely rude and terminated the call without resolving my issue.",
        "I have raised a complaint 4 times and each time I am told it is resolved, but the issue persists.",
        # General Inquiry
        "What documents do I need to submit to increase my credit card limit?",
        "Can you explain how the interest-free period works on my credit card?",
        "Is there a fee for balance transfer from another bank to my credit card account?",
        "How do I register for paperless statements and stop receiving physical mail?",
    ],

    "Neha Gupta": [
        # Fraud
        "My card details were used on a foreign e-commerce website for Rs.23,000 without my consent.",
        "I lost my phone and someone is trying to access my banking app. Please block all access.",
        # Billing
        "I was charged Rs.999 as annual fee which should have been waived for spends above Rs.1 lakh.",
        "My bill shows an EMI conversion charge I never requested.",
        "Interest is being charged on a purchase I returned last month.",
        # Transaction Dispute
        "I paid using UPI but the money was deducted twice for the same transaction.",
        "My card was charged for a cancelled flight booking. The airline confirmed the refund 15 days ago.",
        "A transaction of Rs.18,000 is showing as pending for over a week.",
        # Card Services
        "My contactless payment is not working even though the card supports tap-to-pay.",
        "I want to set a lower daily spending limit on my card for security but cannot find the option.",
        # Account Access
        "I changed my mobile number and am now locked out because OTP goes to the old number.",
        "My net banking password reset link is not being received on my registered email.",
        # Loan & Credit
        "My home loan EMI increased without any prior notice of interest rate revision.",
        "I requested a loan repayment schedule 2 weeks ago but have not received it yet.",
        # Reward & Points
        "My 5x reward points promotion is not being applied on eligible purchases.",
        "I cannot find where to track my milestone bonus on the app.",
        # Customer Service
        "I was on hold for 90 minutes and the call dropped. Nobody called me back.",
        "I sent an email to customer support 10 days ago and have not received any response.",
        # General Inquiry
        "What is the process to convert a large purchase into 0% EMI?",
        "How do I add a nominee to my savings account?",
        "What are the charges for international ATM cash withdrawal?",
        "Can I use my reward points to pay my outstanding credit card bill?",
        "How long does it take to process a credit limit enhancement request?",
    ],

    "Arjun Singh": [
        # Fraud
        "Three consecutive transactions of Rs.15,000 each were made on my card in under 5 minutes.",
        "My card was used at an ATM in a city I have never visited. Please block and investigate.",
        # Billing
        "I was charged a processing fee for a transaction that was clearly declined at the POS.",
        "My statement balance and the outstanding shown in the app are different by Rs.3,400.",
        "Finance charges are continuing to accrue even though I paid the full balance last month.",
        # Transaction Dispute
        "A vendor charged me Rs.11,000 extra due to a POS error. I need a chargeback initiated.",
        "I paid for a product online but the merchant says the payment failed. My account was debited.",
        "A refund of Rs.6,500 from a hotel is overdue by 25 days.",
        # Card Services
        "My card is being declined internationally even though I have international usage enabled.",
        "My add-on card holder's card was lost. I need it blocked without blocking my primary card.",
        # Account Access
        "I cannot view my account balance — the app shows zero for all my accounts erroneously.",
        "My login was successful but all account details are blank. This has been happening for 3 days.",
        # Loan & Credit
        "My loan account shows a different interest rate than what is mentioned in my loan agreement.",
        "I have been paying my loan EMI on time but the outstanding principal is not reducing as expected.",
        # Reward & Points
        "My anniversary bonus of 5,000 points that was promised has not been credited.",
        "Points I earned during the promotional period expired before I could redeem them.",
        # Customer Service
        "The bank executive gave me incorrect information about my loan which cost me extra charges.",
        "I requested a callback 3 days ago and am still waiting.",
        # General Inquiry
        "What is the minimum CIBIL score required to apply for a personal loan?",
        "How do I apply for a secured credit card against my fixed deposit?",
        "What happens if I miss one EMI payment on my personal loan?",
        "Can I prepay part of my personal loan without any penalty?",
        "How does the reward points expiry policy work for inactive accounts?",
    ],

    "Shreya Das": [
        # Fraud
        "I received a phishing email that looked exactly like your official communication. I want to report it.",
        "Someone enrolled a new device on my account without my knowledge.",
        # Billing
        "I was billed for a service fee on my zero-fee account. This contradicts my account terms.",
        "My credit card shows purchases that are higher than what I spent at the merchant.",
        "I have been charged Rs.2,000 for a credit shield insurance I never opted into.",
        # Transaction Dispute
        "My UPI payment of Rs.9,000 failed but the money was deducted and not returned in 5 days.",
        "I did not receive the goods I purchased, and the merchant is not responding. Need chargeback.",
        "A reversal transaction from last month has still not been credited to my account.",
        # Card Services
        "My card number changed after renewal but old EMIs are not being linked to the new card.",
        "I was issued a card with wrong name spelling. I need a replacement immediately.",
        # Account Access
        "My savings account was frozen citing KYC non-compliance but I submitted KYC 3 months ago.",
        "I cannot update my address in the app — the save button is greyed out.",
        # Loan & Credit
        "My credit score report shows a loan I never took. This is fraudulent.",
        "My personal loan was approved but disbursed at a higher interest rate than quoted.",
        # Reward & Points
        "My reward catalogue shows items as available but returns an error when I try to redeem.",
        "Points from my business credit card are not being pooled with my personal card as agreed.",
        # Customer Service
        "The grievance officer has not responded to my written complaint in 15 days.",
        "I was promised a resolution by Friday and it is now the following Wednesday.",
        # General Inquiry
        "What is the process to nominate a family member for my savings account?",
        "How do I dispute a CIBIL entry that I believe is incorrect?",
        "Can I change my credit card billing cycle date?",
        "What documents are needed to apply for a home loan?",
        "How do I register for the mobile banking app for the first time?",
    ],

    "Sandeep Chatterjee": [
        # Fraud
        "I received a message saying my account will be blocked unless I share my card details. Suspicious.",
        "Someone is applying for loans in my name using a stolen identity. Please block all credit activity.",
        # Billing
        "My bill includes a charge for a supplementary card that was cancelled 6 months ago.",
        "I was charged fuel surcharge even though my card has fuel surcharge waiver.",
        "My statement shows a subscription charge that I cancelled a month ago.",
        # Transaction Dispute
        "My net banking transfer of Rs.20,000 was initiated but the beneficiary says money not received.",
        "I paid my electricity bill through the app but the DISCOM shows it as unpaid.",
        "A transaction done by my add-on cardholder is being incorrectly attributed to my account.",
        # Card Services
        "I applied for a business credit card 3 weeks ago and the status still shows under review.",
        "My card was blocked after 3 wrong PIN attempts. I need it unblocked and PIN reset.",
        # Account Access
        "The bank's website is showing an SSL error and I cannot access net banking.",
        "My mobile banking token has expired and I cannot regenerate it without visiting the branch.",
        # Loan & Credit
        "My loan foreclosure amount quoted over the phone is different from what the app shows.",
        "I completed my loan repayment but the NOC has not been issued after 30 days.",
        # Reward & Points
        "I transferred my points to a partner airline but the miles have not appeared.",
        "My reward points statement shows fewer points than I calculated from my purchases.",
        # Customer Service
        "Three different agents gave me three different answers to the same question.",
        "I was asked to visit the branch for an issue that should be resolvable over the phone.",
        # General Inquiry
        "What is the process to convert my credit card dues into a personal loan?",
        "How do I opt out of the credit card insurance programme?",
        "Can I link two savings accounts to a single credit card for autopay?",
        "What is the turnaround time for a credit card dispute resolution?",
        "How do I get a physical copy of my last 12-month bank statement for a visa application?",
    ],

    "Vikram Patel": [
        # Fraud
        "I received an SMS about a transaction I did not make at a store I have never visited.",
        "An unknown app seems to have accessed my saved card details for a subscription.",
        # Billing
        "I was charged Rs.500 for a paper statement when I am enrolled for e-statements.",
        "My outstanding balance includes an unexplained charge of Rs.1,800.",
        "Annual fee was charged on a card I was told is lifetime free.",
        # Transaction Dispute
        "My cashback from a partner merchant has not been credited for the last 2 months.",
        "I overpaid my credit card bill by Rs.5,000. When will the surplus be refunded?",
        "A payment I made at a toll plaza was charged twice.",
        # Card Services
        "My card has been stuck in dispatch status for 3 weeks after being re-issued.",
        "I cannot increase my daily ATM withdrawal limit from the app.",
        # Account Access
        "My account is locked after I failed to complete the video KYC process.",
        "The app is asking me to re-register even though I have been a customer for 3 years.",
        # Loan & Credit
        "I applied for an education loan but have not been contacted by anyone for 2 weeks.",
        "My loan application was rejected without any reason being communicated.",
        # Reward & Points
        "My dining cashback offer did not apply despite spending at a partner restaurant.",
        "I was not notified that my rewards tier was about to be downgraded.",
        # Customer Service
        "I was charged for a service the agent said was free during the call.",
        "The customer care number goes to an IVR that keeps looping without reaching an agent.",
        # General Inquiry
        "What is the difference between a credit card and a charge card?",
        "How do I set up UPI AutoPay for my credit card bill?",
        "Is there a way to get a virtual credit card number for online shopping?",
        "Can I withdraw cash from my credit card at a bank branch instead of ATM?",
        "What happens to my credit card limit if I make a partial payment?",
    ],

    "Anjali Roy": [
        # Fraud
        "I got a call from someone claiming to be your security team asking to verify my CVV.",
        "My card details were used at a merchant in a different country while my card is with me.",
        # Billing
        "I was charged for a reward membership renewal I had explicitly opted out of.",
        "My bill this month has an Rs.800 charge labelled miscellaneous with no description.",
        "I returned goods to a merchant and the credit note is not reflecting on my statement.",
        # Transaction Dispute
        "I cancelled a hotel booking within the free cancellation window but was still charged.",
        "An online subscription charged me in USD and the conversion rate applied seems incorrect.",
        "My refund from an airline cancellation is 40 days overdue.",
        # Card Services
        "I requested a card with a custom image but received a standard card without explanation.",
        "My card was rejected at an international airport duty-free shop.",
        # Account Access
        "After updating my phone, my banking app is asking for full re-registration.",
        "My internet banking access was blocked after a failed login from a new browser.",
        # Loan & Credit
        "My overdraft limit was reduced without any prior communication.",
        "I am being charged a penalty interest rate which should not apply under my loan terms.",
        # Reward & Points
        "A partner merchant's cashback offer was not applied even though I met all conditions.",
        "My Diwali bonus points promotion cashback has not been credited 3 weeks later.",
        # Customer Service
        "I was transferred across 5 agents with no one being able to resolve my issue.",
        "An agent promised to waive my late fee but the waiver was never applied.",
        # General Inquiry
        "How does the rotating cashback category work on my card?",
        "What is the process to temporarily block and then unblock my card?",
        "Can I split a single large transaction into multiple EMIs after the purchase?",
        "How do I update my PAN card number linked to my bank account?",
        "What is the grace period before a late payment is reported to CIBIL?",
    ],

    "Aditi Nair": [
        # Fraud
        "I received a fake bank email asking me to click a link to verify my account. I did not click.",
        "My net banking shows a beneficiary I never added. Someone may have accessed my account.",
        # Billing
        "My credit card interest rate is higher than the rate mentioned in my welcome letter.",
        "I was billed for an overseas transaction fee even though the merchant is India-based.",
        "My statement cycle changed without my consent and it affected my payment schedule.",
        # Transaction Dispute
        "I made an NEFT transfer which failed mid-way. The money has not been returned for 4 days.",
        "I purchased goods that were never delivered and the merchant is unresponsive. Need reversal.",
        "A transaction I made 2 months ago has re-appeared on this month's statement.",
        # Card Services
        "My card PIN change request at the ATM is failing with a generic error.",
        "I have been waiting for my first credit card for 25 days since approval.",
        # Account Access
        "My linked savings account was deactivated without any communication.",
        "I cannot access my account on any device after a security alert email.",
        # Loan & Credit
        "My credit card was downgraded to a lower variant without my consent.",
        "My loan EMI date falls on a weekend and I am being charged a late fee unfairly.",
        # Reward & Points
        "I earned a referral bonus of 2,000 points but it has not been credited after 3 weeks.",
        "My reward points were forfeited when my card was renewed — they should have been carried over.",
        # Customer Service
        "The bank's chatbot is giving incorrect information about my account balance.",
        "I raised a dispute online but the portal shows it was closed without any action.",
        # General Inquiry
        "How do I check if my account is eligible for a pre-approved personal loan?",
        "What is the process to request a refund for a declined transaction?",
        "Can I change my credit card payment due date?",
        "How do I set transaction alerts for amounts above a certain threshold?",
        "Is there a way to freeze specific merchants from charging my card?",
    ],

    "Rahul Sharma": [
        # Fraud
        "Multiple small transactions below Rs.500 are appearing on my statement — looks like card skimming.",
        "Someone added a new device to my account and initiated a fund transfer. Block everything.",
        # Billing
        "I am being charged a higher late payment fee than what is stated in my card's terms.",
        "My billing statement shows a purchase in a currency I did not transact in.",
        "I have been charged Rs.3,500 for a service I cancelled 3 months ago.",
        # Transaction Dispute
        "My payment of Rs.14,000 to a vendor was made twice due to a portal error.",
        "The merchant acknowledged the return but the refund has not appeared after 18 days.",
        "I requested a stop payment on a cheque but it was still processed.",
        # Card Services
        "I need an emergency card replacement as I am travelling internationally next week.",
        "My virtual card is not working for e-commerce transactions.",
        # Account Access
        "My login credentials were compromised. I need all active sessions terminated immediately.",
        "I cannot add a new beneficiary online — the system shows a server error every time.",
        # Loan & Credit
        "My loan closure certificate still shows an outstanding balance of Rs.1.",
        "I have been wrongly marked as a defaulter in your system and need it corrected urgently.",
        # Reward & Points
        "My co-branded card points from airline purchases are not being reflected correctly.",
        "I have enough points to redeem but the catalogue website keeps timing out.",
        # Customer Service
        "I was told my complaint was escalated to the nodal officer but I have heard nothing in 10 days.",
        "The branch manager was unhelpful and dismissed my fraud complaint without proper documentation.",
        # General Inquiry
        "How do I apply for a loan against my fixed deposit?",
        "What is the limit for IMPS transfers through mobile banking?",
        "Can I use my credit card reward points to pay towards my loan EMI?",
        "How do I report suspected account hacking without losing access?",
        "What is the procedure to close my credit card account permanently?",
    ],

    "Priya Menon": [
        # Fraud
        "I have received three suspicious transaction alerts in a row totalling Rs.41,000.",
        "Someone is attempting to reset my banking password from an unknown location.",
        # Billing
        "I was charged an account maintenance fee on an account I was told is zero-balance.",
        "My credit card bill shows duplicate entries for the same transaction.",
        "I requested a bill payment extension but the late fee was still applied.",
        # Transaction Dispute
        "I paid Rs.22,000 for a laptop online but the item was not delivered and seller vanished.",
        "A merchant processed my refund but it has been 3 weeks and nothing credited.",
        "I was charged Rs.1,500 for a membership that auto-renewed which I had cancelled.",
        # Card Services
        "My card was upgraded to a premium variant without my request and I am being charged higher fees.",
        "The card I received has an incorrect expiry date printed on it.",
        # Account Access
        "My account was put under scrutiny and I have not been informed of the reason.",
        "I am unable to change my registered mobile number through the app or branch.",
        # Loan & Credit
        "My car loan balance is showing incorrectly — it is higher than my actual outstanding.",
        "My application for a top-up loan was rejected with no reason given.",
        # Reward & Points
        "Lounge access credits on my card are showing as exhausted but I have only used them twice.",
        "My grocery cashback of Rs.1,200 from last quarter has never been credited.",
        # Customer Service
        "The nodal officer number listed on the website is non-functional.",
        "I sent a registered letter to the grievance department but got no acknowledgement.",
        # General Inquiry
        "How many international lounge visits am I entitled to per quarter on my card?",
        "What is the exchange rate used for foreign currency transactions?",
        "Is there a facility to get a consolidated account statement across all products?",
        "How do I apply for a credit card for my college-going child?",
        "What is the process to change my loan repayment account to a different bank?",
    ],

    "Karan Mehta": [
        # Fraud
        "I received an SMS of Rs.95,000 withdrawal from my account. I am not in the country.",
        "My Platinum card was duplicated and used at luxury stores without my knowledge.",
        # Billing
        "My credit card bill this month is Rs.50,000 higher than my actual spending.",
        "I was charged a forex markup fee on a transaction made in Indian rupees.",
        "My insurance premium linked to the card was debited twice in January.",
        # Transaction Dispute
        "A hotel charged my card in full after I cancelled and the property confirmed the cancellation.",
        "My RTGS payment of Rs.2,00,000 did not reach the beneficiary for 3 days.",
        "The refund for a returned flight ticket is pending for 45 days.",
        # Card Services
        "My supplementary card was blocked but the primary card holder was not notified.",
        "I applied for a metal Platinum card but received the regular plastic variant.",
        # Account Access
        "My account dashboard shows all products as inactive despite normal usage.",
        "I cannot access the premium benefits portal — it says I am not eligible despite my tier.",
        # Loan & Credit
        "My business loan repayment schedule provided by the bank contains an arithmetic error.",
        "I need a loan sanction letter urgently for a property purchase and it has been 2 weeks.",
        # Reward & Points
        "Bonus points from the spend milestone campaign have not been credited after 6 weeks.",
        "Points I earned on my co-branded travel card did not transfer correctly to the airline.",
        # Customer Service
        "My relationship manager changed without any communication and the new RM has not reached out.",
        "I requested priority queue service at the branch but was told it is not available.",
        # General Inquiry
        "What is the maximum amount I can transfer via NEFT in a single day?",
        "How do I apply for a corporate credit card for my employees?",
        "Can I hold multiple credit cards under the same customer ID?",
        "What are the tax benefits on home loan repayment that the bank can certify?",
        "How do I view all active mandates linked to my account?",
    ],

    "Divya Iyer": [
        # Fraud
        "An unknown person has listed my account as a beneficiary and made a transfer.",
        "I received an OTP for a new device registration I did not initiate. Please lock my account.",
        # Billing
        "My credit card has recurring charges from a merchant I cannot identify.",
        "The billing date on my account was changed without my consent affecting my cash flow.",
        "I was charged a 3% transaction fee on a domestic purchase — no such fee was disclosed.",
        # Transaction Dispute
        "I paid Rs.8,000 via the bank's payment gateway but the merchant portal shows payment failed.",
        "My refund from a rejected train ticket booking is 20 days overdue.",
        "Two of my standing instructions processed the same day, causing double deduction.",
        # Card Services
        "My card is showing as expired on some websites even though it does not expire for a year.",
        "I need a card on a different network (RuPay instead of Visa) but no option is shown.",
        # Account Access
        "My biometric login stopped working after a phone software update.",
        "I cannot register for internet banking — the verification call is not coming.",
        # Loan & Credit
        "My education loan EMI deduction has started before the moratorium period ended.",
        "I need a No-Dues Certificate for my closed credit card but have been waiting 3 weeks.",
        # Reward & Points
        "My reward points redemption for movie tickets keeps failing at the last step.",
        "I was promised accelerated points on grocery spends but it has not been applied.",
        # Customer Service
        "An agent enrolled me in a paid service during a resolution call without my permission.",
        "I asked for a written confirmation of my complaint resolution and never received it.",
        # General Inquiry
        "What are the eligibility criteria for the bank's home loan balance transfer scheme?",
        "How do I set up a recurring deposit through the mobile app?",
        "Is there a charge for activating international usage on my debit card?",
        "Can I link my PPF account to my savings account for easy transfers?",
        "What is the maximum number of add-on cards allowed per primary credit card?",
    ],

    "Rohan Verma": [
        # Fraud
        "I found a card reader skimming device at an ATM I recently used. Want to report it.",
        "My debit card details were compromised and used at an e-commerce platform.",
        # Billing
        "I was charged a conversion fee on a purchase that was clearly in Indian rupees.",
        "There is a discrepancy of Rs.2,100 between my credit card app balance and the paper statement.",
        "My zero percent EMI conversion has started charging interest from the second month.",
        # Transaction Dispute
        "My IMPS transfer to a wrong account number. Can you help reverse it?",
        "I purchased a product that was defective but the merchant refuses to refund. Need bank help.",
        "A failed transaction at an ATM deducted cash but did not dispense it.",
        # Card Services
        "My card has been linked to the wrong mobile number for OTP delivery.",
        "The emergency card I requested due to travel has not arrived after 10 days.",
        # Account Access
        "I am unable to log in from abroad — the country appears to be blocked.",
        "I am getting a daily transaction limit error even though I have not hit any limit.",
        # Loan & Credit
        "I paid off my vehicle loan but the hypothecation on the RC has not been removed.",
        "My credit card was closed without my consent after I missed one payment.",
        # Reward & Points
        "My first purchase bonus of 1,000 points was supposed to credit after 60 days and has not.",
        "I cannot find the option to gift reward points to a family member on the app.",
        # Customer Service
        "I was given wrong documentation requirements for my home loan which wasted 2 weeks.",
        "My complaint was closed without resolution and marked as resolved in the system.",
        # General Inquiry
        "How do I apply for an overdraft facility against my salary account?",
        "What is the interest rate applicable for credit card balance transfer?",
        "Can I receive money from abroad directly to my savings account?",
        "How do I activate SMS banking on my account?",
        "What is the procedure to change the primary account holder on a joint account?",
    ],

    "Sunita Reddy": [
        # Fraud
        "I received a call offering a credit limit increase in exchange for my card details. I hung up.",
        "Transactions from international merchants I have never shopped at appear on my statement.",
        # Billing
        "My bill included a charge for a credit card protection plan I have been trying to cancel.",
        "The GST amount charged on my credit card fee appears higher than the applicable rate.",
        "I am being charged interest on a purchase that falls within the interest-free grace period.",
        # Transaction Dispute
        "My transfer to a fixed deposit was debited but the FD was not created.",
        "I paid an advance for a service and the vendor cancelled, but the refund has not come.",
        "I was charged for a failed top-up transaction on a prepaid wallet.",
        # Card Services
        "My Platinum card does not provide lounge access as promised in the benefits guide.",
        "I want to downgrade my card to avoid annual fees but the option is unavailable.",
        # Account Access
        "I moved abroad and cannot complete the annual KYC update through the app.",
        "My account access was restricted citing AML review, and I was not informed.",
        # Loan & Credit
        "My loan was transferred to a collection agency even though I had made a payment arrangement.",
        "I applied for a credit card top-up loan but the funds were disbursed to a wrong account.",
        # Reward & Points
        "I enrolled in the spend-based bonus programme but the milestone reward was not given.",
        "My points account shows a negative balance which makes no sense.",
        # Customer Service
        "I was promised a senior agent callback within 24 hours and it has been 4 days.",
        "The bank's social media team responded to my complaint but took no actual action.",
        # General Inquiry
        "How do I opt in to receive balance alerts via WhatsApp instead of SMS?",
        "What is the process to convert a fixed deposit to a recurring deposit?",
        "Can I use my NRE account to invest in Indian mutual funds?",
        "How do I link my Aadhaar to my savings account through the app?",
        "What is the maximum cash deposit allowed per day at a branch without PAN?",
    ],

    "Manish Joshi": [
        # Fraud
        "I believe my card number was stolen from a data breach. Multiple small test transactions appeared.",
        "Unauthorized standing instructions were added to my account without my knowledge.",
        # Billing
        "I was billed for a service that was supposed to be included free in my Platinum membership.",
        "My statement shows a processing fee deducted for a loan I applied for but never received.",
        "I was charged interest from the date of purchase instead of the due date.",
        # Transaction Dispute
        "My cheque payment of Rs.35,000 was bounced by the bank even though funds were available.",
        "An NACH debit was processed twice in the same month for the same loan.",
        "A transaction that I successfully reversed is appearing again on this month's statement.",
        # Card Services
        "My virtual card keeps getting declined even on websites I have previously used it on.",
        "I need a card with a higher daily spending limit for my business purchases.",
        # Account Access
        "I cannot access my NRI banking portal since returning to India.",
        "My account shows dormant status even though I transacted last week.",
        # Loan & Credit
        "My top-up loan was approved but disbursed after the property registration date I needed it for.",
        "I foreclosed my personal loan but was charged a penalty that should not apply per RBI guidelines.",
        # Reward & Points
        "I was told my points expire in March but the app now shows they expired in January.",
        "My spend on the bank's own investment products is not earning reward points as promised.",
        # Customer Service
        "The escalation matrix on the bank's website leads to email IDs that bounce.",
        "I filed a complaint with the Banking Ombudsman and the bank has not responded within 30 days.",
        # General Inquiry
        "What is the best way to consolidate multiple credit card dues into a single loan?",
        "How do I update the nominee details on my fixed deposits?",
        "Can I get a statement of all my financial products together in one PDF?",
        "What is the RBI-mandated turnaround time for resolving a credit card dispute?",
        "How do I opt out of receiving pre-approved loan offers?",
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
