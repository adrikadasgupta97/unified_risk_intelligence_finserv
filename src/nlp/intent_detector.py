"""
Rule-augmented zero-shot intent detector.
Uses keyword patterns as primary signal; falls back to zero-shot classification
for ambiguous inputs when a HuggingFace classifier is available.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src.config import config

INTENT_PATTERNS: dict[str, list[str]] = {
    "register_complaint": [
        r"\b(lodge|file|register|submit|raise|report)\b.*(complaint|issue|problem|concern)",
        r"\b(i want to|i need to|i'd like to).*(complain|report|raise|lodge|file)",
        r"\b(this is unacceptable|very (frustrated|unhappy|upset|angry))",
        r"\b(charged|debited|billed).*(wrong|incorrectly|extra|twice|unauthorized)",
        r"\b(fraud|scam|unauthorized).*(transaction|charge|activity)",
        r"\b(my (card|account|transaction|payment)).*(not working|blocked|failed|declined|error)",
        r"\b(called|contacted|emailed).*(support|service|helpline).*(times|again|multiple)\b",
        r"\b(called|contacted).*(times|again|multiple|repeatedly)\b",
        r"\b(no one|nobody).*(resolved|helped|called back|responded|addressed)\b",
        r"\b(still|yet).*(not|un)(resolved|fixed|addressed|responded)\b",
        r"\b(not|not yet|still not).*(resolved|fixed|sorted|addressed|closed)\b",
        r"\b(issue|problem|complaint).*(is not|isn.t|has not|hasn.t|not been|not yet).*(resolved|fixed|closed|sorted)\b",
        r"\bit.*(is not|isn.t|not yet|still).*(resolved|fixed|sorted|done)\b",
        r"\b(hasn'?t|haven'?t|not been|never been).*(resolved|fixed|sorted|closed|addressed)\b",
        r"\b(charge|fee|penalty|interest).*(applied|deducted|charged|levied).*(even though|despite|although)\b",
        r"\b(applied|deducted|charged|levied).*(even though|despite|although|but)\b",
        r"\b(without|without any).*(notice|notification|communication|prior|my knowledge|my consent)\b",
        r"\b(not (received|credited|reflected|updated|responded|processed))\b",
        r"\b(still (haven.?t|not|pending|waiting|no))\b",
        r"\b(my (emi|loan|credit|card|account|payment|points|cashback|refund|otp|limit|statement|balance))\b",
        r"\b(otp|one.time password).*(not received|didn.?t receive|not getting|not coming)\b",
        r"\b(locked out|suspended|blocked|declined|failed|bounce|bounced)\b",
        r"\b(incorrect|wrong|error|mismatch).*(balance|amount|charge|statement|limit|rate)\b",
        r"\b(reduce[d]?|cut|decreased).*(credit limit|limit)\b",
        r"\b(reward points?|cashback|refund).*(not credited|not received|missing|pending|expired)\b",
        r"\b(replacement card|renewed card|new card).*(not received|haven.?t received|still waiting)\b",
        r"\b(waiting|been waiting|no response|no update|no callback).*(days|weeks|hours)\b",
        r"\b(rude|unprofessional|disconnected|hung up).*(agent|representative|executive)\b",
        r"\b(on hold).*(minutes|hours)\b",
        r"\b(promised|said).*(call back|resolve|fix).*(never|didn.t|has not|haven.t)\b",
    ],
    "track_complaint": [
        r"\b(status|update|progress|where).*(complaint|ticket|case|issue)",
        r"\b(track|check|follow up).*(complaint|ticket|request)",
        r"\bcomplaint.*(id|number|ref|reference)\b",
        r"\b(CMP-[A-Z0-9]+)\b",
        r"\bhow long.*(resolution|resolve|fix)",
    ],
    "resolve_query": [
        r"\b(how (do|can|to)|what is|explain|help me).*(reward|point|limit|fee|interest|statement)",
        r"\b(what (is|are)|tell me about).*(benefit|feature|offer|charges)",
        r"\b(i want to know|i have a question)\b",
    ],
    "escalate_to_agent": [
        r"\b(speak|talk|connect|transfer).*(agent|human|person|representative|manager|supervisor)",
        r"\b(escalate|raise to (manager|supervisor))\b",
        r"\b(i (want|need|would like) to (speak|talk)).*(someone|person|human)\b",
    ],
    "close_complaint": [
        r"\b(issue|problem|complaint|it).*(has been|is now|was|is).*(resolved|fixed|sorted|closed)\b",
        r"\b(close|resolve|mark).*(complaint|ticket|case|issue)\b",
        r"\b(complaint|issue|ticket).*(close|closed|resolved|done|sorted|fixed)\b",
        r"\b(i want to|i need to|please|can you).*(close|resolve|mark).*(complaint|issue|ticket|this)\b",
        r"\b(thank you|thanks|thank u|thx).*(help|resolving|fixing|sorting|addressing)\b",
        r"\b(thank you|thanks|thank u|thx).*(for (your|the) (help|support|assistance))\b",
        r"\b(thank you|thanks).*(issue|problem|concern)\b",
        r"\bno (more|further).*(issue|problem|concern)\b",
        r"\bproblem (solved|fixed|gone|sorted)\b",
        r"\b(complaint|issue|problem) (resolved|fixed|closed|sorted|done)\b",
        r"\b(resolved|fixed|sorted|closed)[\s!.]*$",
        r"\b(issue|problem).*(is|has been).*(taken care of|sorted|fixed|resolved)\b",
        r"\b(all (good|fine|ok|okay)|everything('s| is) (fine|good|resolved|sorted))\b",
    ],
    "greet": [
        r"^(hi|hello|hey|good (morning|afternoon|evening)|howdy)[\s!.,]*$",
        r"\b(start|begin|initiate)\b",
    ],
    "goodbye": [
        r"^(bye|goodbye|see you|that('s| is) all|done|exit|quit)[\s!.,]*$",
    ],
}

_COMPILED: dict[str, list[re.Pattern]] = {
    intent: [re.compile(p, re.IGNORECASE) for p in patterns]
    for intent, patterns in INTENT_PATTERNS.items()
}


@dataclass
class IntentResult:
    intent: str
    confidence: float
    method: str  # "pattern" or "zeroshot"


_NEGATION_RE = re.compile(
    r"\b(not|no|never|neither|hasn'?t|haven'?t|didn'?t|don'?t|doesn'?t|isn'?t|wasn'?t|yet|still|no one|nobody)\b",
    re.IGNORECASE,
)

_CLOSE_RESOLUTION_RE = re.compile(r"\b(resolved|fixed|sorted|closed)\b", re.IGNORECASE)


def _has_negated_resolution(text: str) -> bool:
    """Return True if the text says something is NOT resolved (e.g. 'not yet resolved')."""
    if not _CLOSE_RESOLUTION_RE.search(text):
        return False
    # Find position of resolution word; check for negation within 5 words before it
    for m in _CLOSE_RESOLUTION_RE.finditer(text):
        preceding = text[:m.start()]
        preceding_words = preceding.split()[-5:]
        if any(_NEGATION_RE.search(w) for w in preceding_words):
            return True
    return False


def detect_intent(text: str) -> IntentResult:
    text_clean = text.strip()

    # If message contains a negated resolution phrase, never route to close_complaint
    negated = _has_negated_resolution(text_clean)

    # Pattern matching — fast and interpretable
    matches: list[tuple[str, int]] = []
    for intent, patterns in _COMPILED.items():
        if negated and intent == "close_complaint":
            continue
        count = sum(1 for p in patterns if p.search(text_clean))
        if count:
            matches.append((intent, count))

    if matches:
        best_intent, count = max(matches, key=lambda x: x[1])
        confidence = min(0.65 + count * 0.10, 0.95)
        return IntentResult(intent=best_intent, confidence=confidence, method="pattern")

    # If no pattern matched and zero-shot classifier is available, use it
    try:
        result = _zeroshot_classify(text_clean, exclude_intents={"close_complaint"} if negated else None)
        if result and result.confidence >= config.nlp.intent_confidence_threshold:
            return result
    except Exception:
        pass

    return IntentResult(intent="register_complaint", confidence=0.40, method="fallback")


def _zeroshot_classify(text: str, exclude_intents: set = None) -> Optional[IntentResult]:
    from transformers import pipeline

    labels = [l for l in INTENT_PATTERNS.keys() if not (exclude_intents and l in exclude_intents)]
    classifier = _get_zeroshot_pipeline()
    out = classifier(text, candidate_labels=labels, multi_label=False)
    top_label = out["labels"][0]
    top_score = float(out["scores"][0])
    return IntentResult(intent=top_label, confidence=top_score, method="zeroshot")


_zeroshot_pipeline = None


def _get_zeroshot_pipeline():
    global _zeroshot_pipeline
    if _zeroshot_pipeline is None:
        from transformers import pipeline
        _zeroshot_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,
        )
    return _zeroshot_pipeline
