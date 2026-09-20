"""
Customer chat UI — complaint registration, tracking and automated resolution.
Only accessible to users with role='customer'.
"""
import re as _re
import uuid
import streamlit as st
from src.security.auth import change_password, TokenData
from src.chatbot.conversation_manager import ConversationManager
from src.data.database import init_db

init_db()

# Guard: customers only
if not st.session_state.get("authenticated") or st.session_state.customer.role != "customer":
    st.error("Access denied.")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_models():
    placeholder = st.empty()

    placeholder.info("🧠 Teaching our AI to understand your frustration...")
    from src.nlp.sentiment_analyzer import _get_pipeline
    _get_pipeline()

    placeholder.info("🤝 Convincing our AI that the customer is always right...")
    from src.nlp.entity_extractor import _get_spacy
    _get_spacy()

    placeholder.info("☕ Almost ready — good things take a moment...")
    from src.rag.retriever import retrieve_relevant_articles
    retrieve_relevant_articles("test", top_k=1)

    placeholder.empty()
    return True


load_models()

# ------------------------------------------------------------------ #
# Session state                                                        #
# ------------------------------------------------------------------ #
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "conversation_manager" not in st.session_state:
    st.session_state.conversation_manager = ConversationManager()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "greeted" not in st.session_state:
    st.session_state.greeted = False
if "secondary_docs_map" not in st.session_state:
    st.session_state.secondary_docs_map = {}

# ------------------------------------------------------------------ #
# Sidebar                                                              #
# ------------------------------------------------------------------ #
with st.sidebar:
    customer: TokenData = st.session_state.customer
    st.success(f"Logged in as **{customer.name}**")
    st.caption(f"Tier: {customer.customer_value_tier}")

    with st.expander("Change Password"):
        cp_current = st.text_input("Current Password", type="password", key="cp_current")
        cp_new = st.text_input("New Password", type="password", key="cp_new")
        cp_confirm = st.text_input("Confirm New Password", type="password", key="cp_confirm")
        if st.button("Update Password"):
            if not all([cp_current, cp_new, cp_confirm]):
                st.error("All fields are required.")
            elif cp_new != cp_confirm:
                st.error("New passwords do not match.")
            elif len(cp_new) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                result = change_password(customer.customer_id, cp_current, cp_new)
                if result.success:
                    st.success("Password changed successfully.")
                else:
                    st.error(result.message)

    if st.button("Logout", use_container_width=True):
        for key in ["authenticated", "customer", "session_id", "conversation_manager",
                    "chat_history", "greeted", "secondary_docs_map"]:
            st.session_state.pop(key, None)
        st.rerun()

# ------------------------------------------------------------------ #
# Chat UI                                                              #
# ------------------------------------------------------------------ #
st.title("💬 Financial Services Complaint Assistant")
st.caption("Powered by NLP · RAG · AI")


def _render_secondary_docs(docs: list[dict]):
    for doc in docs:
        body = doc["content"].strip()
        lines = body.splitlines()
        if lines and lines[0].strip() == doc["title"].strip():
            body = "\n".join(lines[1:]).strip()
        body = _re.sub(r'\s*\((\d+)\)\s*', lambda m: f"\n{m.group(1)}. ", body).strip()
        with st.expander(f"📄 {doc['title']}"):
            st.markdown(body)


cm: ConversationManager = st.session_state.conversation_manager

if not st.session_state.greeted:
    greeting = cm.start_session(st.session_state.customer, st.session_state.session_id)
    st.session_state.chat_history.append({"role": "assistant", "content": greeting})
    st.session_state.greeted = True

for idx, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
    if msg["role"] == "assistant" and idx in st.session_state.secondary_docs_map:
        _render_secondary_docs(st.session_state.secondary_docs_map[idx])

if user_input := st.chat_input("Type your complaint or question here..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Processing your request..."):
            response = cm.handle_message(st.session_state.session_id, user_input)
        st.markdown(response.message)

    msg_idx = len(st.session_state.chat_history)
    st.session_state.chat_history.append({"role": "assistant", "content": response.message})

    if response.secondary_docs:
        st.session_state.secondary_docs_map[msg_idx] = response.secondary_docs
        _render_secondary_docs(response.secondary_docs)
