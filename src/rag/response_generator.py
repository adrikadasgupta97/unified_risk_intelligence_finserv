"""
Generates context-aware mitigation responses from retrieved KB articles.

Primary: template-based formatter using top retrieved KB article (no API needed, instant).
Optional: Claude LLM generation when API credits are available (set USE_LLM=true in .env).
"""
from __future__ import annotations

import os
import textwrap

from src.config import config
from src.rag.retriever import RetrievedDocument

_USE_LLM = os.getenv("USE_LLM", "false").lower() == "true"


def generate_mitigation_response(
    complaint_text: str,
    retrieved_docs: list[RetrievedDocument],
    category: str = "",
    customer_name: str = "valued customer",
) -> str:
    if not retrieved_docs:
        return (
            f"Thank you for reaching out, {customer_name}. I understand your concern. "
            "I was unable to find a matching resolution in our knowledge base for your specific issue. "
            "I will escalate this to a senior support agent who will contact you shortly. "
            "Your complaint has been registered and you will receive a reference number."
        )

    if _USE_LLM:
        try:
            return _llm_response(complaint_text, retrieved_docs, category, customer_name)
        except Exception:
            pass  # Fall through to template on any API error

    return _template_response(retrieved_docs, customer_name)


def _template_response(docs: list[RetrievedDocument], customer_name: str) -> str:
    import re

    top = docs[0]

    # Extract content body, dropping a leading line that repeats the title
    content_lines = top.content.strip().splitlines()
    if content_lines and content_lines[0].strip() == top.title.strip():
        content_lines = content_lines[1:]
    body = "\n".join(content_lines).strip()

    # Split inline numbered steps like "(1) Do this. (2) Do that." into separate lines
    body = re.sub(r'\s*\((\d+)\)\s*', lambda m: f"\n{m.group(1)}. ", body).strip()

    response = (
        f"Thank you for reaching out, {customer_name}. "
        f"I'm sorry to hear about this issue. Here's how we can help:\n\n"
        f"**{top.title}**\n\n"
        f"{body}"
    )

    return response


def _llm_response(
    complaint_text: str,
    docs: list[RetrievedDocument],
    category: str,
    customer_name: str,
) -> str:
    import anthropic

    _SYSTEM_PROMPT = textwrap.dedent("""
        You are a helpful and empathetic customer support assistant for a financial services company.
        Provide clear, accurate, and actionable mitigation steps based ONLY on the provided knowledge
        base articles. Be concise and empathetic. Use numbered steps. Keep responses under 200 words.
        Do not reveal internal scores or categories.
    """).strip()

    kb_context = "\n\n---\n\n".join(
        f"[{doc.title}]\n{doc.content}" for doc in docs
    )
    user_message = (
        f"Customer complaint: {complaint_text}\n\n"
        f"Relevant knowledge base articles:\n{kb_context}\n\n"
        f"Please provide a helpful response to resolve this {category} complaint."
    )

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=config.llm.model,
        max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return resp.content[0].text.strip()
