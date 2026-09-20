"""
Train an ML risk priority classifier using synthetic training data.

Training strategy: generate a large, balanced grid of risk factor combinations,
call compute_risk_score() on each to obtain rule-based priority labels (pseudo-labels),
then train RF and GBM classifiers to learn those relationships.

Why useful:
  - ML discovers which features matter most through training (no hand-coded weights needed).
  - Feature importances validate (or challenge) the domain expert's hand-coded factor weights.
  - With real priority labels from human reviewers, the ML model would supersede the rule-based system.

Features (6, all numeric):
  severity_base        — category-specific base severity from _SEVERITY_BASE
  sentiment_component  — (1 − polarity) / 2  ∈ [0, 1]
  escalation_norm      — min(escalation_count / 3, 1)
  unresolved_norm      — min(unresolved_count / 5, 1)
  freq_norm            — min(freq_30d / 10, 1)
  tier_multiplier      — customer value multiplier from _CUSTOMER_VALUE_MULTIPLIER

Output: Priority string — Critical / High / Medium / Low

Usage:
    python scripts/train_risk_scorer.py
"""
import sys
import random
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

from src.classification.risk_scorer import (
    compute_risk_score,
    _SEVERITY_BASE,
    _CUSTOMER_VALUE_MULTIPLIER,
    _MAX_ESCALATIONS,
    _MAX_UNRESOLVED,
    _MAX_FREQUENCY,
)

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "risk_scorer.pkl"

FEATURE_NAMES = [
    "severity_base",
    "sentiment_component",
    "escalation_norm",
    "unresolved_norm",
    "freq_norm",
    "tier_multiplier",
]

CATEGORIES = list(_SEVERITY_BASE.keys())
TIERS = list(_CUSTOMER_VALUE_MULTIPLIER.keys())


# Low-severity categories that can reach Low priority with positive sentiment
_LOW_PRIORITY_CATS = ["General Inquiry", "Reward & Points Issue", "Customer Service Complaint"]


def generate_dataset(n_train: int = 2000, n_test: int = 400, seed: int = 42):
    """Generate rule-labelled training and test sets by sampling feature combinations.

    Uses use_ml=False so pseudo-labels always come from the rule-based scorer,
    avoiding circular dependency with any previously trained model.

    Low priority is rare in pure random sampling (requires low severity + positive
    sentiment + no escalation + low frequency + Standard tier simultaneously).
    We force-inject ~200 Low samples to ensure all 4 classes are represented.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    def _sample(force_low: bool = False):
        if force_low:
            cat  = rng.choice(_LOW_PRIORITY_CATS)
            pol  = round(rng.uniform(0.4, 1.0), 3)   # positive sentiment
            esc  = 0
            unr  = 0
            freq = rng.randint(1, 3)
            tier = "Standard"
        else:
            cat  = rng.choice(CATEGORIES)
            pol  = round(rng.uniform(-1.0, 1.0), 3)
            esc  = rng.randint(0, _MAX_ESCALATIONS)
            unr  = rng.randint(0, _MAX_UNRESOLVED)
            freq = rng.randint(1, _MAX_FREQUENCY)
            tier = rng.choice(TIERS)

        result = compute_risk_score(
            category=cat, complaint_text="",
            sentiment_polarity=pol,
            escalation_count=esc,
            unresolved_count=unr,
            complaint_frequency_30d=freq,
            customer_value_tier=tier,
            use_ml=False,  # always use rule-based labels; never the ML model itself
        )
        row = [
            _SEVERITY_BASE.get(cat, 0.30),
            (1.0 - pol) / 2.0,
            min(esc  / _MAX_ESCALATIONS, 1.0),
            min(unr  / _MAX_UNRESOLVED,  1.0),
            min(freq / _MAX_FREQUENCY,   1.0),
            _CUSTOMER_VALUE_MULTIPLIER.get(tier, 0.50),
        ]
        return row, result.priority.value

    # Reserve 200 slots in training for forced Low samples (fills up to that many)
    N_FORCE_LOW = 200
    train_rows, train_labels = [], []
    forced_low = 0
    for i in range(n_train):
        force = forced_low < N_FORCE_LOW and i < N_FORCE_LOW * 2
        r, lbl = _sample(force_low=force)
        if force:
            forced_low += 1
        train_rows.append(r)
        train_labels.append(lbl)

    # Test set: pure random (no forced Low — tests real distribution)
    test_rows, test_labels = [], []
    for _ in range(n_test):
        r, lbl = _sample(force_low=False)
        test_rows.append(r)
        test_labels.append(lbl)

    return (
        np.array(train_rows), np.array(train_labels),
        np.array(test_rows),  np.array(test_labels),
    )


def main():
    print("=" * 65)
    print("ML Risk Scorer Training — RF vs Gradient Boosting")
    print("=" * 65)

    X_train, y_train, X_test, y_test = generate_dataset()
    print(f"\nTraining samples: {len(X_train)}")
    print(f"  Label distribution: {dict(Counter(y_train))}")
    print(f"Test samples: {len(X_test)}")
    print(f"  Label distribution: {dict(Counter(y_test))}")

    model_factories = {
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        ),
        "GradientBoosting": lambda: GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results: dict[str, dict] = {}
    best_acc = -1.0
    best_name = ""
    best_clf = None

    for name, factory in model_factories.items():
        print(f"\n--- {name} ---")
        clf = factory()
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
        print(f"  5-fold CV Accuracy: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

        clf.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"  Test Accuracy:      {test_acc*100:.1f}%")
        print(classification_report(y_test, clf.predict(X_test),
                                    labels=["Critical", "High", "Medium", "Low"],
                                    zero_division=0))

        cv_results[name] = {
            "cv_mean":  round(float(scores.mean()), 4),
            "cv_std":   round(float(scores.std()),  4),
            "test_acc": round(float(test_acc),       4),
        }
        if test_acc > best_acc:
            best_acc = test_acc
            best_name = name
            best_clf  = clf

    print(f"\n{'='*65}")
    print(f"Best model: {best_name}  (test accuracy {best_acc*100:.1f}%)")

    # Feature importances
    importances = best_clf.feature_importances_
    ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    print(f"\n--- Feature Importances ({best_name}) ---")
    for feat_name, imp in ranked:
        print(f"  {feat_name:<25} {imp*100:>6.1f}%")

    MODELS_DIR.mkdir(exist_ok=True)
    bundle = {
        "model":          best_clf,
        "model_name":     best_name,
        "feature_names":  FEATURE_NAMES,
        "cv_results":     cv_results,
        "test_accuracy":  round(best_acc, 4),
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved {best_name} to {MODEL_PATH}")


if __name__ == "__main__":
    main()
