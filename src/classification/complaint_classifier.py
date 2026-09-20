"""
Complaint category classifier.
Tier 1: keyword/pattern rules (fast, interpretable).
Tier 2: trained TF-IDF + Logistic Regression (when models/category_classifier.pkl exists).
Tier 3: zero-shot classification via BART-MNLI for ambiguous inputs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.data.models import ComplaintCategory

_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "category_classifier.pkl"


_CATEGORY_PATTERNS: dict[str, list[str]] = {
    ComplaintCategory.FRAUD: [
        r"\b(fraud|scam|unauthorized|suspicious)\b",
        r"\b(did not (make|authorize|initiate))\b",
        r"\b(stolen|compromised|hacked)\b.*(card|account)",
    ],
    ComplaintCategory.TRANSACTION_DISPUTE: [
        r"\b(wrong|incorrect|double|duplicate|extra|overcharged)\b.*(charge|debit|transaction|amount)",
        r"\b(charged|deducted).*(twice|double|wrong amount)",
        r"\b(transaction|payment|transfer).*(failed|not (received|credited|reflected|processed))\b",
        r"\b(refund|reversal|chargeback)\b",
    ],
    ComplaintCategory.ACCOUNT_ACCESS: [
        r"\b(cannot|can't|unable to|not able to).*(log in|login|access|sign in|open)\b",
        r"\b(account|card).*(blocked|locked|suspended|disabled|frozen)\b",
        r"\b(password|pin|otp).*(not (received|working)|expired|wrong|issue)\b",
    ],
    ComplaintCategory.BILLING: [
        r"\b(bill|statement|invoice|due|minimum payment|emi)\b",
        r"\b(late fee|penalty|surcharge|interest charge)\b",
        r"\b(wrong (bill|statement|amount due))\b",
    ],
    ComplaintCategory.REWARDS: [
        r"\b(reward|points?|cashback|miles?|offer|discount|voucher)\b",
        r"\b(not (credited|received|applied|reflected))\b.*(reward|point|cashback)",
        r"\b(redeem|redemption).*(failed|issue|not working)\b",
    ],
    ComplaintCategory.CARD_SERVICES: [
        r"\b(new card|replacement card|card delivery|card not (received|arrived|delivered))\b",
        r"\b(card (damaged|expired|stolen))\b",
        r"\b(increase|decrease).*(credit limit|limit)\b",
        r"\b(virtual card|contactless|tap to pay|nfc)\b",
    ],
    ComplaintCategory.LOAN_CREDIT: [
        r"\b(loan|emi|credit limit|pre-approved|personal loan|home loan|auto loan)\b",
        r"\b(interest rate|apr|processing fee).*(loan|credit)\b",
    ],
    ComplaintCategory.CUSTOMER_SERVICE: [
        r"\b(agent|executive|representative|staff|employee).*(rude|unhelpful|not responding|ignored)\b",
        r"\b(poor|bad|terrible|worst).*(service|support|experience|response)\b",
        r"\b(no response|not responding|waiting|hold|escalate)\b",
    ],
}

_COMPILED: dict[str, list[re.Pattern]] = {
    cat: [re.compile(p, re.IGNORECASE) for p in patterns]
    for cat, patterns in _CATEGORY_PATTERNS.items()
}

_ZS_LABELS = [c.value for c in ComplaintCategory]


@dataclass
class ClassificationResult:
    category: ComplaintCategory
    confidence: float
    method: str


def classify_complaint(text: str) -> ClassificationResult:
    scores: list[tuple[str, int]] = []
    for category, patterns in _COMPILED.items():
        hits = sum(1 for p in patterns if p.search(text))
        if hits:
            scores.append((category, hits))

    if scores:
        best_cat, hits = max(scores, key=lambda x: x[1])
        pattern_conf = min(0.60 + hits * 0.12, 0.95)

        # Soft ensemble: when pattern fires with only 1 hit (low confidence),
        # consult the trained model and take the higher-confidence result.
        if hits == 1:
            trained = _get_trained_pipeline()
            if trained is not None:
                try:
                    model_label = trained.predict([text])[0]
                    model_conf  = float(max(trained.predict_proba([text])[0]))
                    if model_conf > pattern_conf:
                        return ClassificationResult(
                            category=ComplaintCategory(model_label),
                            confidence=model_conf,
                            method="ensemble",
                        )
                except Exception:
                    pass

        return ClassificationResult(
            category=ComplaintCategory(best_cat),
            confidence=pattern_conf,
            method="pattern",
        )

    # Tier 2: trained TF-IDF + LR sklearn model (when models/category_classifier.pkl exists)
    trained = _get_trained_pipeline()
    if trained is not None:
        try:
            label = trained.predict([text])[0]
            proba = trained.predict_proba([text])[0]
            return ClassificationResult(
                category=ComplaintCategory(label),
                confidence=float(max(proba)),
                method="trained",
            )
        except Exception:
            pass

    # Tier 3: zero-shot BART-MNLI
    try:
        return _zeroshot_classify(text)
    except Exception:
        return ClassificationResult(
            category=ComplaintCategory.GENERAL,
            confidence=0.35,
            method="fallback",
        )


def _zeroshot_classify(text: str) -> ClassificationResult:
    pipe = _get_zeroshot_pipeline()
    out = pipe(text, candidate_labels=_ZS_LABELS, multi_label=False)
    top_label = out["labels"][0]
    top_score = float(out["scores"][0])
    return ClassificationResult(
        category=ComplaintCategory(top_label),
        confidence=top_score,
        method="zeroshot",
    )


_trained_pipeline = None
_trained_pipeline_loaded = False


def _get_trained_pipeline():
    global _trained_pipeline, _trained_pipeline_loaded
    if not _trained_pipeline_loaded:
        _trained_pipeline_loaded = True
        if _MODEL_PATH.exists():
            import joblib
            _trained_pipeline = joblib.load(_MODEL_PATH)
    return _trained_pipeline


_zs_pipeline = None


# ------------------------------------------------------------------ #
# XAI helpers                                                         #
# ------------------------------------------------------------------ #

def get_class_top_terms(class_label: str, n: int = 10) -> list[tuple[str, float]]:
    """
    Returns the top n word n-grams most predictive of `class_label`
    from the LR coefficients — independent of any specific input text.
    """
    pipeline = _get_trained_pipeline()
    if pipeline is None:
        return []
    try:
        import numpy as np
        classes = list(pipeline.classes_)
        if class_label not in classes:
            return []
        class_idx = classes.index(class_label)
        coef = pipeline.named_steps["clf"].coef_[class_idx]

        fu = pipeline.named_steps["features"]
        word_names = fu.transformer_list[0][1].get_feature_names_out()
        n_word = len(word_names)

        word_coef = coef[:n_word]
        top_idx = np.argsort(word_coef)[::-1][:n]
        return [(str(word_names[i]), round(float(word_coef[i]), 4)) for i in top_idx]
    except Exception:
        return []


def get_top_features(text: str, n: int = 5) -> tuple[str, list[tuple[str, float]]]:
    """
    Returns (predicted_label, top_n contributing word n-grams) for `text`.
    Contributions = TF-IDF weight × LR coefficient for the predicted class.
    """
    pipeline = _get_trained_pipeline()
    if pipeline is None:
        return ("", [])
    try:
        import numpy as np
        pred_label = pipeline.predict([text])[0]
        classes = list(pipeline.classes_)
        class_idx = classes.index(pred_label)

        X = pipeline.named_steps["features"].transform([text])
        coef = pipeline.named_steps["clf"].coef_[class_idx]
        contribs = X.toarray().flatten() * coef

        fu = pipeline.named_steps["features"]
        word_names = fu.transformer_list[0][1].get_feature_names_out()
        n_word = len(word_names)

        word_contribs = contribs[:n_word]
        pos_mask = word_contribs > 0
        if not pos_mask.any():
            return (str(pred_label), [])
        pos_idx = np.where(pos_mask)[0]
        top_idx = pos_idx[np.argsort(word_contribs[pos_idx])[::-1][:n]]
        return (
            str(pred_label),
            [(str(word_names[i]), round(float(word_contribs[i]), 4)) for i in top_idx],
        )
    except Exception:
        return ("", [])


def _get_zeroshot_pipeline():
    global _zs_pipeline
    if _zs_pipeline is None:
        from transformers import pipeline
        _zs_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,
        )
    return _zs_pipeline
