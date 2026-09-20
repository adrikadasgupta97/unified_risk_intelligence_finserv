"""
Train a TF-IDF + Logistic Regression category classifier on complaint data from SQLite.
Saves the fitted pipeline to models/category_classifier.pkl.

Training data: all complaint texts in SQLite with non-null categories.
Test data:     data/eval/test_categories.json (45 hand-labelled samples).

Usage:
    python scripts/train_category_classifier.py
"""
import sys
import json
import sqlite3
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")

import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from src.data.database import DB_PATH

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "category_classifier.pkl"
EVAL_FILE  = "data/eval/test_categories.json"
SUPP_FILE  = "data/supplementary_train_categories.json"

LABELS_ORDER = [
    "Fraud & Unauthorized Charges", "Billing Concern", "Transaction Dispute",
    "Account Access Issue", "Card Services", "Reward & Points Issue",
    "Loan & Credit Issue", "Customer Service Complaint", "General Inquiry",
]


def load_training_data() -> tuple[list[str], list[str]]:
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT anonymized_text, category FROM complaints "
        "WHERE category IS NOT NULL AND anonymized_text IS NOT NULL"
    ).fetchall()
    conn.close()
    texts  = [r[0] for r in rows]
    labels = [r[1] for r in rows]

    # Merge supplementary hand-crafted samples for underrepresented classes
    if Path(SUPP_FILE).exists():
        with open(SUPP_FILE) as f:
            supp = json.load(f)
        for s in supp:
            texts.append(s["text"])
            labels.append(s["label"])

    return texts, labels


def load_eval_data() -> tuple[list[str], list[str]]:
    with open(EVAL_FILE) as f:
        samples = json.load(f)
    return [s["text"] for s in samples], [s["label"] for s in samples]


def main():
    print("=" * 60)
    print("Category Classifier Training")
    print("=" * 60)

    X_train, y_train = load_training_data()
    print(f"\nTraining samples: {len(X_train)}")
    dist = Counter(y_train)
    for lbl in LABELS_ORDER:
        print(f"  {lbl:<40} {dist.get(lbl, 0):>4}")

    # FeatureUnion: word n-grams (1-2) + character n-grams (3-5) for better short-text coverage
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            max_features=10000, sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            max_features=8000, sublinear_tf=True,
        )),
    ])
    pipeline = Pipeline([
        ("features", features),
        ("clf",      LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    print("Model trained.")

    X_test, y_test = load_eval_data()
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=LABELS_ORDER, zero_division=0)

    print(f"\n--- Evaluation on {len(X_test)}-sample hand-labelled test set ---")
    print(f"Accuracy: {acc*100:.1f}%  (zero-shot baseline: 64.4%)")
    print(f"\n{report}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
