"""
Entry point — login page and role-based navigation router.
Run: streamlit run app.py
"""
import uuid
import streamlit as st
from src.data.database import init_db
from src.security.auth import authenticate_customer, register_customer, reset_password, lookup_role_by_email

init_db()

st.set_page_config(
    page_title="Financial Services – Complaint Agent",
    page_icon="💬",
    layout="wide",
)

# ------------------------------------------------------------------ #
# Session state                                                        #
# ------------------------------------------------------------------ #
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "customer" not in st.session_state:
    st.session_state.customer = None
if "fp_role" not in st.session_state:
    st.session_state.fp_role = None
if "fp_email_confirmed" not in st.session_state:
    st.session_state.fp_email_confirmed = ""

# ------------------------------------------------------------------ #
# Role-based navigation                                                #
# ------------------------------------------------------------------ #
if st.session_state.authenticated:
    role = st.session_state.customer.role
    if role == "admin":
        pages = [
            st.Page("pages/analytics.py",  title="Analytics Dashboard", icon="📊"),
            st.Page("pages/evaluation.py", title="System Evaluation",   icon="🧪"),
        ]
    else:
        pages = [st.Page("pages/chat.py", title="Chat Assistant", icon="💬")]

    pg = st.navigation(pages)
    pg.run()
    st.stop()

# ------------------------------------------------------------------ #
# Login / Register (shown only when not authenticated)                #
# ------------------------------------------------------------------ #
st.title("💬 Financial Services Complaint Assistant")
st.caption("Powered by NLP · RAG · AI")
st.divider()

login_tab, reg_tab, forgot_tab = st.tabs(["Login", "Register", "Forgot Password"])

_LOGIN_OVERLAY = """
<style>
#login-overlay {
    position: fixed; inset: 0;
    background: rgba(255,255,255,0.97);
    z-index: 9999;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 18px;
}
#login-overlay .logo {
    font-size: 72px; line-height: 1;
    animation: pulse 1.4s ease-in-out infinite;
}
#login-overlay .msg {
    font-size: 20px; font-weight: 600;
    color: #065A82; letter-spacing: 0.3px;
}
#login-overlay .sub {
    font-size: 13px; color: #888;
}
@keyframes pulse {
    0%, 100% { transform: scale(1);   opacity: 1;   }
    50%       { transform: scale(1.1); opacity: 0.8; }
}
</style>
<div id="login-overlay">
    <div class="logo">💬</div>
    <div class="msg">Logging you in…</div>
    <div class="sub">Verifying your credentials securely</div>
</div>
"""

with login_tab:
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_pass")
    if st.button("Login", use_container_width=True):
        if not email or not password:
            st.error("Please enter your email and password.")
        else:
            overlay = st.empty()
            overlay.markdown(_LOGIN_OVERLAY, unsafe_allow_html=True)
            result = authenticate_customer(email, password)
            overlay.empty()
            if result.success:
                st.session_state.authenticated = True
                st.session_state.customer = result.customer
                st.rerun()
            else:
                st.error(result.message)

with forgot_tab:
    # Step 1 — email lookup
    if not st.session_state.fp_role:
        st.markdown("Enter your registered email to get started.")
        fp_email = st.text_input("Registered Email", key="fp_email")
        if st.button("Continue", use_container_width=True):
            if not fp_email:
                st.error("Please enter your email.")
            else:
                role = lookup_role_by_email(fp_email)
                if role:
                    st.session_state.fp_role = role
                    st.session_state.fp_email_confirmed = fp_email
                    st.rerun()
                else:
                    st.error("No account found with that email.")

    # Step 2 — role-specific verification + new password
    else:
        role = st.session_state.fp_role
        email = st.session_state.fp_email_confirmed
        st.info(f"Verifying identity for **{email}**")

        if role == "admin":
            verify_input = st.text_input("Full Name (as registered)", key="fp_name")
        else:
            verify_input = st.text_input("Account Number", key="fp_acc")

        fp_new = st.text_input("New Password", type="password", key="fp_new")
        fp_confirm = st.text_input("Confirm New Password", type="password", key="fp_confirm")

        col1, col2 = st.columns(2)
        if col1.button("Reset Password", use_container_width=True):
            if not all([verify_input, fp_new, fp_confirm]):
                st.error("All fields are required.")
            elif fp_new != fp_confirm:
                st.error("Passwords do not match.")
            elif len(fp_new) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                kwargs = {"name": verify_input} if role == "admin" else {"account_number": verify_input}
                result = reset_password(email, fp_new, **kwargs)
                if result.success:
                    st.success(result.message)
                    st.session_state.fp_role = None
                    st.session_state.fp_email_confirmed = ""
                else:
                    st.error(result.message)
        if col2.button("Back", use_container_width=True):
            st.session_state.fp_role = None
            st.session_state.fp_email_confirmed = ""
            st.rerun()

with reg_tab:
    r_name = st.text_input("Full Name")
    r_email = st.text_input("Email", key="reg_email")
    r_acc = st.text_input("Account Number")
    r_tier = st.selectbox("Account Tier", ["Standard", "Gold", "Platinum"])
    r_role = st.selectbox("Role", ["customer", "admin"])
    r_pass = st.text_input("Password", type="password", key="reg_pass")
    if st.button("Register", use_container_width=True):
        if not all([r_name, r_email, r_acc, r_pass]):
            st.error("All fields are required.")
        else:
            cid = f"CUST-{uuid.uuid4().hex[:6].upper()}"
            ok = register_customer(cid, r_name, r_email, r_acc, r_pass, r_tier, r_role)
            if ok:
                st.success(f"Registered! Your ID: **{cid}**. Please login.")
            else:
                st.error("Email or account number already registered.")
