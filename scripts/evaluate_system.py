# -*- coding: utf-8 -*-
"""
System Evaluation Script
Measures: Intent Detection, Category Classification, Sentiment Analysis, RAG Retrieval
Outputs results to data/eval/results.json
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import json
import time
sys.path.insert(0, ".")

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

from src.nlp.intent_detector import detect_intent
from src.nlp.sentiment_analyzer import analyze_sentiment
from src.rag.retriever import retrieve_relevant_articles

DATA_DIR = "data/eval"
RESULTS_FILE = f"{DATA_DIR}/results.json"


def load(filename):
    with open(f"{DATA_DIR}/{filename}") as f:
        return json.load(f)


def report_dict(y_true, y_pred, labels):
    """Return per-class and overall metrics as a dict."""
    report = classification_report(y_true, y_pred, labels=labels,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    acc = round(accuracy_score(y_true, y_pred), 4)
    return {"accuracy": acc, "report": report, "confusion_matrix": cm, "labels": labels}


# ------------------------------------------------------------------ #
# 1. Intent Detection                                                  #
# ------------------------------------------------------------------ #
def eval_intent():
    print("\n--- Intent Detection ---")
    samples = load("test_intents.json")
    y_true, y_pred = [], []
    labels_order = ["register_complaint", "track_complaint", "close_complaint",
                    "escalate_to_agent", "resolve_query", "greet", "goodbye"]
    for i, s in enumerate(samples):
        result = detect_intent(s["text"])
        y_true.append(s["label"])
        y_pred.append(result.intent)
        status = "✓" if result.intent == s["label"] else f"✗ (got {result.intent})"
        print(f"  [{i+1:02d}] {status} | {s['text'][:60]}")
    metrics = report_dict(y_true, y_pred, labels_order)
    print(f"\n  Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"  Macro F1: {metrics['report']['macro avg']['f1-score']*100:.1f}%")
    return metrics


# ------------------------------------------------------------------ #
# 2. Category Classification                                           #
# ------------------------------------------------------------------ #
def eval_categories():
    """Evaluate pure zero-shot BART-MNLI only (no pattern matching, no trained model)."""
    print("\n--- Category Classification (Zero-shot baseline) ---")
    from src.classification.complaint_classifier import _zeroshot_classify
    samples = load("test_categories.json")
    y_true, y_pred = [], []
    labels_order = [
        "Fraud & Unauthorized Charges", "Billing Concern", "Transaction Dispute",
        "Account Access Issue", "Card Services", "Reward & Points Issue",
        "Loan & Credit Issue", "Customer Service Complaint", "General Inquiry",
    ]
    for i, s in enumerate(samples):
        result = _zeroshot_classify(s["text"])
        predicted = result.category.value
        y_true.append(s["label"])
        y_pred.append(predicted)
        status = "✓" if predicted == s["label"] else f"✗ (got {predicted})"
        print(f"  [{i+1:02d}] {status} | {s['text'][:60]}")
    metrics = report_dict(y_true, y_pred, labels_order)
    print(f"\n  Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"  Macro F1: {metrics['report']['macro avg']['f1-score']*100:.1f}%")
    return metrics


# ------------------------------------------------------------------ #
# 2b. Category Classification — Trained Model                         #
# ------------------------------------------------------------------ #
def eval_categories_trained():
    """Evaluate the trained TF-IDF + LR model (Tier 2) on the same 45-sample test set."""
    from pathlib import Path
    if not Path("models/category_classifier.pkl").exists():
        print("\n--- Category Classification (Trained) --- SKIPPED (model not found)")
        return None

    print("\n--- Category Classification (Trained Model) ---")
    samples = load("test_categories.json")
    y_true, y_pred = [], []
    labels_order = [
        "Fraud & Unauthorized Charges", "Billing Concern", "Transaction Dispute",
        "Account Access Issue", "Card Services", "Reward & Points Issue",
        "Loan & Credit Issue", "Customer Service Complaint", "General Inquiry",
    ]
    import joblib
    pipeline = joblib.load("models/category_classifier.pkl")
    for i, s in enumerate(samples):
        predicted = pipeline.predict([s["text"]])[0]
        y_true.append(s["label"])
        y_pred.append(predicted)
        status = "✓" if predicted == s["label"] else f"✗ (got {predicted})"
        print(f"  [{i+1:02d}] {status} | {s['text'][:60]}")
    metrics = report_dict(y_true, y_pred, labels_order)
    print(f"\n  Accuracy: {metrics['accuracy']*100:.1f}%  (zero-shot baseline: 64.4%)")
    print(f"  Macro F1: {metrics['report']['macro avg']['f1-score']*100:.1f}%")
    return metrics


# ------------------------------------------------------------------ #
# 2c. Category Classification — DistilBERT fine-tuned                 #
# ------------------------------------------------------------------ #
def eval_categories_distilbert():
    from pathlib import Path
    model_path = Path("models/distilbert_category")
    if not model_path.exists():
        print("\n--- Category Classification (DistilBERT) --- SKIPPED (model not found)")
        return None

    print("\n--- Category Classification (DistilBERT Fine-tuned) ---")
    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline("text-classification", model=str(model_path), device=-1)
    samples = load("test_categories.json")
    labels_order = [
        "Fraud & Unauthorized Charges", "Billing Concern", "Transaction Dispute",
        "Account Access Issue", "Card Services", "Reward & Points Issue",
        "Loan & Credit Issue", "Customer Service Complaint", "General Inquiry",
    ]
    y_true, y_pred = [], []
    for i, s in enumerate(samples):
        out = pipe(s["text"], truncation=True, max_length=128)
        predicted = out[0]["label"]
        y_true.append(s["label"])
        y_pred.append(predicted)
        status = "✓" if predicted == s["label"] else f"✗ (got {predicted})"
        print(f"  [{i+1:02d}] {status} | {s['text'][:60]}")
    metrics = report_dict(y_true, y_pred, labels_order)
    print(f"\n  Accuracy: {metrics['accuracy']*100:.1f}%  (TF-IDF+LR: 84.4%,  zero-shot: 64.4%)")
    print(f"  Macro F1: {metrics['report']['macro avg']['f1-score']*100:.1f}%")
    return metrics


# ------------------------------------------------------------------ #
# 3. Sentiment Analysis                                                #
# ------------------------------------------------------------------ #
def eval_sentiment():
    print("\n--- Sentiment Analysis ---")
    samples = load("test_sentiments.json")
    y_true, y_pred = [], []
    labels_order = ["negative", "neutral", "positive"]
    for i, s in enumerate(samples):
        result = analyze_sentiment(s["text"])
        predicted = result.label.value
        y_true.append(s["label"])
        y_pred.append(predicted)
        status = "✓" if predicted == s["label"] else f"✗ (got {predicted})"
        print(f"  [{i+1:02d}] {status} | {s['text'][:60]}")
    metrics = report_dict(y_true, y_pred, labels_order)
    print(f"\n  Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"  Macro F1: {metrics['report']['macro avg']['f1-score']*100:.1f}%")
    return metrics


# ------------------------------------------------------------------ #
# 4. RAG Retrieval Relevance                                          #
# ------------------------------------------------------------------ #
def eval_rag():
    print("\n--- RAG Retrieval Relevance ---")
    test_cases = [
        {"query": "My reward points have not been credited after purchase",
         "expected_category": "Reward & Points Issue", "expected_keywords": ["reward", "points", "cashback"]},
        {"query": "I cannot log into my mobile banking app after changing phone",
         "expected_category": "Account Access Issue", "expected_keywords": ["login", "access", "account", "otp"]},
        {"query": "There is an unauthorized transaction on my credit card",
         "expected_category": "Fraud & Unauthorized Charges", "expected_keywords": ["fraud", "unauthorized", "transaction", "card"]},
        {"query": "I was charged a late fee even though I paid on time",
         "expected_category": "Billing Concern", "expected_keywords": ["fee", "payment", "charge", "billing"]},
        {"query": "How do I redeem my reward points for flight vouchers?",
         "expected_category": "Reward & Points Issue", "expected_keywords": ["redeem", "reward", "points", "voucher"]},
        {"query": "I want to foreclose my personal loan early",
         "expected_category": "Loan & Credit Issue", "expected_keywords": ["loan", "foreclose", "prepay", "emi"]},
        {"query": "My replacement credit card has not arrived after 3 weeks",
         "expected_category": "Card Services", "expected_keywords": ["card", "replacement", "delivery", "block"]},
        {"query": "My UPI payment failed but amount was deducted from account",
         "expected_category": "Transaction Dispute", "expected_keywords": ["transaction", "payment", "refund", "deduct"]},
        {"query": "How do I set up automatic payment for my credit card bill?",
         "expected_category": "General Inquiry", "expected_keywords": ["autopay", "automatic", "payment", "bill"]},
        {"query": "I want to report a phishing attempt asking for my card details",
         "expected_category": "Fraud & Unauthorized Charges", "expected_keywords": ["fraud", "phishing", "scam", "security"]},
    ]

    hit_at_1 = 0
    hit_at_3 = 0

    for i, tc in enumerate(test_cases):
        docs = retrieve_relevant_articles(tc["query"])
        if not docs:
            print(f"  [{i+1:02d}] NO RESULTS | {tc['query'][:55]}")
            continue

        top1_content = (docs[0].title + " " + docs[0].content).lower()
        top3_content = " ".join((d.title + " " + d.content).lower() for d in docs[:3])

        h1 = any(kw in top1_content for kw in tc["expected_keywords"])
        h3 = any(kw in top3_content for kw in tc["expected_keywords"])

        if h1: hit_at_1 += 1
        if h3: hit_at_3 += 1

        status = "✓" if h1 else ("~" if h3 else "✗")
        print(f"  [{i+1:02d}] {status} Top doc: '{docs[0].title[:45]}' | {tc['query'][:40]}")

    n = len(test_cases)
    precision_at_1 = round(hit_at_1 / n, 4)
    precision_at_3 = round(hit_at_3 / n, 4)
    print(f"\n  Precision@1: {precision_at_1*100:.1f}%")
    print(f"  Precision@3: {precision_at_3*100:.1f}%")
    return {
        "precision_at_1": precision_at_1,
        "precision_at_3": precision_at_3,
        "n_test_cases": n,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
    }


# ------------------------------------------------------------------ #
# 5. Risk Scoring Sanity Check                                         #
# ------------------------------------------------------------------ #
def eval_risk_scoring():
    print("\n--- Risk Scoring Sanity Check ---")
    from src.classification.risk_scorer import compute_risk_score

    test_cases = [
        # (category, sentiment_polarity, tier, unresolved, freq, expected_priority)
        ("Fraud & Unauthorized Charges", -0.9, "Platinum", 5, 8, "Critical"),
        ("Fraud & Unauthorized Charges", -0.7, "Gold",     3, 5, "High"),
        ("Billing Concern",              -0.5, "Standard", 2, 3, "High"),
        ("Billing Concern",              -0.1, "Standard", 0, 1, "Medium"),
        ("General Inquiry",               0.2, "Standard", 0, 1, "Low"),
        ("Customer Service Complaint",   -0.8, "Platinum", 4, 6, "Critical"),
        ("Transaction Dispute",          -0.3, "Gold",     1, 2, "Medium"),
        ("Reward & Points Issue",        -0.2, "Standard", 0, 1, "Low"),
    ]

    correct = 0
    results = []
    for cat, polarity, tier, unresolved, freq, expected in test_cases:
        r = compute_risk_score(
            category=cat, complaint_text="test",
            sentiment_polarity=polarity,
            escalation_count=0, unresolved_count=unresolved,
            complaint_frequency_30d=freq, customer_value_tier=tier,
            use_ml=False,
        )
        got = r.priority.value
        ok = got == expected
        if ok: correct += 1
        status = "✓" if ok else f"✗ (got {got})"
        print(f"  {status} | {cat[:35]} | {tier:<10} | score={r.total_score:.2f} | expected={expected}")
        results.append({"category": cat, "tier": tier, "expected": expected,
                        "predicted": got, "score": round(r.total_score, 4)})

    acc = round(correct / len(test_cases), 4)
    print(f"\n  Priority Accuracy: {acc*100:.1f}%")
    return {"accuracy": acc, "n_test_cases": len(test_cases), "correct": correct, "cases": results}


# ------------------------------------------------------------------ #
# 5b. Risk Scoring — ML model                                          #
# ------------------------------------------------------------------ #
def eval_risk_scoring_ml():
    from pathlib import Path
    if not Path("models/risk_scorer.pkl").exists():
        print("\n--- Risk Scoring (ML) --- SKIPPED (model not found)")
        return None

    print("\n--- Risk Scoring (ML Model) ---")
    import joblib, numpy as np
    from src.classification.risk_scorer import (
        _SEVERITY_BASE, _CUSTOMER_VALUE_MULTIPLIER,
        _MAX_ESCALATIONS, _MAX_UNRESOLVED, _MAX_FREQUENCY,
    )
    bundle = joblib.load("models/risk_scorer.pkl")
    model  = bundle["model"]

    test_cases = [
        ("Fraud & Unauthorized Charges", -0.9, "Platinum", 5, 8, "Critical"),
        ("Fraud & Unauthorized Charges", -0.7, "Gold",     3, 5, "High"),
        ("Billing Concern",              -0.5, "Standard", 2, 3, "High"),
        ("Billing Concern",              -0.1, "Standard", 0, 1, "Medium"),
        ("General Inquiry",               0.2, "Standard", 0, 1, "Low"),
        ("Customer Service Complaint",   -0.8, "Platinum", 4, 6, "Critical"),
        ("Transaction Dispute",          -0.3, "Gold",     1, 2, "Medium"),
        ("Reward & Points Issue",        -0.2, "Standard", 0, 1, "Low"),
    ]

    correct = 0
    results = []
    for cat, polarity, tier, unresolved, freq, expected in test_cases:
        row = np.array([[
            _SEVERITY_BASE.get(cat, 0.30),
            (1.0 - polarity) / 2.0,
            min(0 / _MAX_ESCALATIONS, 1.0),
            min(unresolved / _MAX_UNRESOLVED, 1.0),
            min(freq / _MAX_FREQUENCY, 1.0),
            _CUSTOMER_VALUE_MULTIPLIER.get(tier, 0.50),
        ]])
        got = str(model.predict(row)[0])
        ok  = got == expected
        if ok: correct += 1
        status = "✓" if ok else f"✗ (got {got})"
        print(f"  {status} | {cat[:35]} | {tier:<10} | expected={expected}")
        results.append({"category": cat, "tier": tier, "expected": expected, "predicted": got})

    acc = round(correct / len(test_cases), 4)
    model_name = bundle.get("model_name", "ML")
    print(f"\n  Priority Accuracy ({model_name}): {acc*100:.1f}%")
    return {
        "accuracy":     acc,
        "model_name":   model_name,
        "n_test_cases": len(test_cases),
        "correct":      correct,
        "cases":        results,
        "cv_results":   bundle.get("cv_results", {}),
        "test_accuracy_training": bundle.get("test_accuracy", None),
    }


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("=" * 60)
    print("SYSTEM EVALUATION — Intelligent Complaint Management Agent")
    print("=" * 60)
    t0 = time.time()

    trained_cat    = eval_categories_trained()
    distilbert_cat = eval_categories_distilbert()
    results = {
        "intent_detection":                    eval_intent(),
        "category_classification":             eval_categories(),
        "category_classification_trained":     trained_cat,
        "category_classification_distilbert":  distilbert_cat,
        "sentiment_analysis":                  eval_sentiment(),
        "rag_retrieval":                       eval_rag(),
        "risk_scoring":                        eval_risk_scoring(),
        "risk_scoring_ml":                     eval_risk_scoring_ml(),
    }

    elapsed = round(time.time() - t0, 1)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Intent Detection     Accuracy: {results['intent_detection']['accuracy']*100:.1f}%  "
          f"Macro-F1: {results['intent_detection']['report']['macro avg']['f1-score']*100:.1f}%")
    print(f"  Category Classif.   Accuracy: {results['category_classification']['accuracy']*100:.1f}%  "
          f"Macro-F1: {results['category_classification']['report']['macro avg']['f1-score']*100:.1f}%")
    if results.get("category_classification_trained"):
        t = results["category_classification_trained"]
        print(f"  Category (TF-IDF)   Accuracy: {t['accuracy']*100:.1f}%  "
              f"Macro-F1: {t['report']['macro avg']['f1-score']*100:.1f}%")
    if results.get("category_classification_distilbert"):
        d = results["category_classification_distilbert"]
        print(f"  Category (DistilBERT) Accuracy: {d['accuracy']*100:.1f}%  "
              f"Macro-F1: {d['report']['macro avg']['f1-score']*100:.1f}%")
    print(f"  Sentiment Analysis  Accuracy: {results['sentiment_analysis']['accuracy']*100:.1f}%  "
          f"Macro-F1: {results['sentiment_analysis']['report']['macro avg']['f1-score']*100:.1f}%")
    print(f"  RAG Precision@1:    {results['rag_retrieval']['precision_at_1']*100:.1f}%  "
          f"Precision@3: {results['rag_retrieval']['precision_at_3']*100:.1f}%")
    print(f"  Risk Scoring (rule) Accuracy: {results['risk_scoring']['accuracy']*100:.1f}%")
    if results.get("risk_scoring_ml"):
        ml = results["risk_scoring_ml"]
        print(f"  Risk Scoring (ML)   Accuracy: {ml['accuracy']*100:.1f}%  model={ml['model_name']}")
    print(f"\n  Evaluation completed in {elapsed}s")

    results["elapsed_seconds"] = elapsed

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE}")
