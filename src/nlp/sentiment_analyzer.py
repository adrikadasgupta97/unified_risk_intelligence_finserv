"""
Sentiment analyzer using FinBERT (financial domain BERT).
Returns sentiment label and a [-1, 1] polarity score used in risk scoring.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.data.models import SentimentLabel


@dataclass
class SentimentResult:
    label: SentimentLabel
    score: float        # raw confidence for the predicted label
    polarity: float     # [-1.0 (very negative) .. +1.0 (very positive)]


_MODEL_NAME = "ProsusAI/finbert"
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "text-classification",
            model=_MODEL_NAME,
            tokenizer=_MODEL_NAME,
            top_k=None,
            device=-1,
            truncation=True,
            max_length=512,
        )
    return _pipeline


def analyze_sentiment(text: str) -> SentimentResult:
    pipe = _get_pipeline()
    results: list[dict] = pipe(text)[0]   # list of {label, score}

    scores = {r["label"].lower(): r["score"] for r in results}
    pos = scores.get("positive", 0.0)
    neg = scores.get("negative", 0.0)
    neu = scores.get("neutral", 0.0)

    # Polarity: +1 fully positive, -1 fully negative
    polarity = round(pos - neg, 4)

    if neg > pos and neg > neu:
        label = SentimentLabel.NEGATIVE
        score = neg
    elif pos > neg and pos > neu:
        label = SentimentLabel.POSITIVE
        score = pos
    else:
        label = SentimentLabel.NEUTRAL
        score = neu

    return SentimentResult(label=label, score=round(score, 4), polarity=polarity)
