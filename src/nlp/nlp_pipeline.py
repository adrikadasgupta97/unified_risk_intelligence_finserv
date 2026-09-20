"""
Orchestrates the full NLP pipeline: intent → entities → sentiment.
Returns a single NLPOutput object consumed by downstream components.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.nlp.intent_detector import IntentResult, detect_intent
from src.nlp.entity_extractor import ExtractedEntities, extract_entities
from src.nlp.sentiment_analyzer import SentimentResult, analyze_sentiment


@dataclass
class NLPOutput:
    intent: IntentResult
    entities: ExtractedEntities
    sentiment: SentimentResult

    def to_complaint_fields(self) -> dict:
        return {
            "intent": self.intent.intent,
            "entities": self.entities.to_dict(),
            "sentiment": self.sentiment.label.value,
            "sentiment_score": self.sentiment.polarity,
        }


def run_nlp_pipeline(text: str) -> NLPOutput:
    intent = detect_intent(text)
    entities = extract_entities(text)
    sentiment = analyze_sentiment(text)
    return NLPOutput(intent=intent, entities=entities, sentiment=sentiment)
