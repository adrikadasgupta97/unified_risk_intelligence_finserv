"""
Manages multi-turn conversational state for the chatbot.
Routes user input to the appropriate agent action based on detected intent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Optional

from src.chatbot.agent import ComplaintAgent, AgentResponse
from src.nlp.intent_detector import detect_intent
from src.security.auth import TokenData


@dataclass
class ConversationState:
    customer: TokenData
    session_id: str
    history: list[dict] = field(default_factory=list)   # [{role, content}]
    last_complaint_id: Optional[str] = None


class ConversationManager:
    def __init__(self):
        self._agent = ComplaintAgent()
        self._sessions: dict[str, ConversationState] = {}

    def start_session(self, customer: TokenData, session_id: str) -> str:
        self._sessions[session_id] = ConversationState(
            customer=customer,
            session_id=session_id,
        )
        return (
            f"Hello, {customer.name}! Welcome to the Financial Services Support Assistant. "
            "I can help you with:\n"
            "- **Register** a new complaint\n"
            "- **Track** an existing complaint\n"
            "- **Get help** with account or card issues\n"
            "- **Speak with an agent** if needed\n\n"
            "How can I assist you today?"
        )

    def handle_message(self, session_id: str, user_message: str) -> AgentResponse:
        state = self._sessions.get(session_id)
        if not state:
            return AgentResponse(message="Session expired. Please log in again.")

        state.history.append({"role": "user", "content": user_message})

        intent_result = detect_intent(user_message)
        intent = intent_result.intent

        response = self._route(intent, user_message, state)

        state.history.append({"role": "assistant", "content": response.message})
        return response

    def _route(
        self,
        intent: str,
        user_message: str,
        state: ConversationState,
    ) -> AgentResponse:
        customer = state.customer

        if intent == "greet":
            return AgentResponse(
                message=f"Hello again, {customer.name}! How can I help you today?"
            )

        if intent == "goodbye":
            return AgentResponse(
                message=f"Thank you for contacting us, {customer.name}. Have a great day! "
                        "Your complaint reference numbers have been saved for your records."
            )

        if intent == "register_complaint":
            # Collect unresolved count from DB for accurate risk scoring
            from src.data.database import get_customer_complaints
            existing = get_customer_complaints(customer.customer_id)
            unresolved = sum(1 for c in existing if c.status not in ("Resolved", "Closed"))
            frequency_30d = len(existing)

            response = self._agent.register_complaint(
                customer_id=customer.customer_id,
                complaint_text=user_message,
                customer_name=customer.name,
                customer_value_tier=customer.customer_value_tier,
                unresolved_count=unresolved,
                complaint_frequency_30d=frequency_30d,
            )
            if response.complaint_id:
                state.last_complaint_id = response.complaint_id
                response.message = (
                    f"Your complaint has been registered.\n"
                    f"**Reference ID: {response.complaint_id}**\n\n"
                    + response.message
                )
            return response

        if intent == "track_complaint":
            # Try to extract complaint ID from message, else use last registered
            match = re.search(r"\bCMP-[A-Z0-9]{6,10}\b", user_message)
            complaint_id = match.group(0) if match else state.last_complaint_id

            if not complaint_id:
                return AgentResponse(
                    message="Please provide your complaint reference number (e.g., CMP-XXXXXXXX) "
                            "so I can look up the status for you."
                )
            return self._agent.track_complaint(complaint_id, customer.customer_id)

        if intent == "escalate_to_agent":
            if state.last_complaint_id:
                return self._agent.escalate_complaint(
                    state.last_complaint_id, customer.customer_id
                )
            return AgentResponse(
                message="I'll connect you with a senior support agent right away. "
                        "Please hold while I transfer you. Typical wait time is under 5 minutes."
            )

        if intent == "close_complaint":
            if state.last_complaint_id:
                return self._agent.resolve_complaint(state.last_complaint_id)
            return AgentResponse(
                message="I'm glad your issue has been resolved! "
                        "If you have a complaint reference number, I can formally close it for you."
            )

        if intent == "resolve_query":
            # General informational question — retrieve KB answer without registering a complaint
            from src.rag.retriever import retrieve_relevant_articles
            from src.rag.response_generator import generate_mitigation_response
            retrieved = retrieve_relevant_articles(user_message)
            if retrieved:
                answer = generate_mitigation_response(
                    complaint_text=user_message,
                    retrieved_docs=retrieved,
                    category="General Inquiry",
                    customer_name=customer.name,
                )
                secondary = [{"title": d.title, "content": d.content} for d in retrieved[1:]]
                return AgentResponse(message=answer, secondary_docs=secondary)
            return AgentResponse(
                message=f"I'd be happy to help, {customer.name}. Could you provide more details about your query?"
            )

        # Default: treat as a new complaint and register it
        return self._agent.register_complaint(
            customer_id=customer.customer_id,
            complaint_text=user_message,
            customer_name=customer.name,
            customer_value_tier=customer.customer_value_tier,
        )
