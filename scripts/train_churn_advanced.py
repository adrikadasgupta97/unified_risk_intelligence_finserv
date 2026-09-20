"""
Advanced churn classifier training: compares Random Forest vs Gradient Boosting via LOOCV.
Saves the best-performing model (with full comparison metadata) to models/churn_classifier.pkl,
replacing the previous single-RF bundle.

Usage:
    python scripts/train_churn_advanced.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report

from src.analytics.churn_analyzer import compute_churn_report, _WEIGHTS, _tier

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "churn_classifier.pkl"

FEATURE_NAMES = [
    "complaint_frequency_30d", "complaint_frequency_90d",
    "avg_sentiment_polarity",  "sentiment_trend",
    "escalation_rate",         "unresolved_rate",
    "avg_risk_score",          "max_risk_score",
    "critical_count",          "days_since_last_complaint",
]


def features_to_row(f) -> list[float]:
    return [
        float(f.complaint_frequency_30d), float(f.complaint_frequency_90d),
        float(f.avg_sentiment_polarity),  float(f.sentiment_trend),
        float(f.escalation_rate),         float(f.unresolved_rate),
        float(f.avg_risk_score),          float(f.max_risk_score),
        float(f.critical_count),          float(f.days_since_last_complaint),
    ]


def loocv_eval(clf_factory, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Run LOOCV and return (accuracy, per-fold predictions)."""
    loo = LeaveOneOut()
    y_pred = np.empty(len(y), dtype=object)
    for train_idx, test_idx in loo.split(X):
        clf = clf_factory()
        clf.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = clf.predict(X[test_idx])
    return float(accuracy_score(y, y_pred)), y_pred


def main():
    print("=" * 65)
    print("Advanced Churn Classifier — Random Forest vs Gradient Boosting")
    print("=" * 65)

    report = compute_churn_report()
    customers = report.customers
    if len(customers) < 3:
        print("Not enough customer data to train (need ≥3 customers).")
        return

    print(f"\nCustomers: {len(customers)}")
    print(f"Tier distribution: {dict(Counter(c.churn_risk for c in customers))}")

    X = np.array([features_to_row(c) for c in customers])
    # Always derive labels from the rule-based score threshold, not from _predict_tier()
    # which may use a stale RF model that predicts only a subset of tiers.
    y = np.array([_tier(c.churn_score) for c in customers])

    model_factories = {
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        ),
        "GradientBoosting": lambda: GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
        ),
    }

    loocv_results: dict[str, dict] = {}
    best_acc = -1.0
    best_name = ""

    for name, factory in model_factories.items():
        print(f"\n--- {name} LOOCV ---")
        acc, y_pred = loocv_eval(factory, X, y)
        print(f"  LOOCV Accuracy: {acc*100:.1f}%")
        print(classification_report(y, y_pred, labels=["High", "Medium", "Low"], zero_division=0))
        loocv_results[name] = {"accuracy": round(acc, 4), "y_pred": y_pred.tolist()}
        if acc > best_acc:
            best_acc = acc
            best_name = name

    print(f"\n{'='*65}")
    print(f"Best model: {best_name}  ({best_acc*100:.1f}% LOOCV accuracy)")

    # Fit best model on full dataset
    best_clf = model_factories[best_name]()
    best_clf.fit(X, y)

    # Feature importances
    importances = best_clf.feature_importances_
    ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    print(f"\n--- Feature Importances ({best_name}) vs Hand-coded Weights ---")
    print(f"  {'Feature':<35} {'ML Importance':>14}  {'Rule Weight':>11}")
    print("  " + "-" * 65)
    for feat_name, imp in ranked:
        rule_w = _WEIGHTS.get(feat_name, 0.0)
        print(f"  {feat_name:<35} {imp*100:>12.1f}%  {rule_w*100:>10.1f}%")

    MODELS_DIR.mkdir(exist_ok=True)
    bundle = {
        "model":          best_clf,
        "model_name":     best_name,
        "feature_names":  FEATURE_NAMES,
        "loocv_results":  {k: {"accuracy": v["accuracy"]} for k, v in loocv_results.items()},
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved {best_name} to {MODEL_PATH}")


if __name__ == "__main__":
    main()
