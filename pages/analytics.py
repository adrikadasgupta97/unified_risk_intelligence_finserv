"""
Admin Analytics Dashboard — complaint classification and risk scoring insights.
Only accessible to users with role='admin'.
"""
import sqlite3
import json
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.database import DB_PATH
from src.analytics.churn_analyzer import compute_churn_report

# Guard: admins only
if not st.session_state.get("authenticated") or st.session_state.customer.role != "admin":
    st.error("Access denied.")
    st.stop()

st.title("📊 Financial Services Complaint Analytics Dashboard")
st.caption("Complaint classification, risk scoring and sentiment insights")

with st.sidebar:
    st.success(f"Logged in as **{st.session_state.customer.name}**")
    st.caption("Role: Admin")
    if st.button("Logout", use_container_width=True):
        for key in ["authenticated", "customer"]:
            st.session_state.pop(key, None)
        st.rerun()


# ------------------------------------------------------------------ #
# Colour palette — single source of truth                             #
# ------------------------------------------------------------------ #
# Semantic colours used consistently everywhere:
#   Red    → bad / high severity / negative sentiment / critical
#   Orange → warning / medium severity
#   Green  → good / low severity / positive / resolved
#   Blue   → neutral informational bars and lines (no severity meaning)
#   Grey   → neutral sentiment label

C_RED    = "#C0392B"
C_ORANGE = "#E67E22"
C_GREEN  = "#27AE60"
C_BLUE   = "#2980B9"
C_GREY   = "#7F8C8D"

# Severity maps — used wherever a value carries explicit risk meaning
PRIORITY_COLORS  = {"Critical": C_RED, "High": C_ORANGE, "Medium": C_BLUE, "Low": C_GREEN}
SENTIMENT_COLORS = {"negative": C_RED, "neutral": C_BLUE,  "positive": C_GREEN}
CHURN_COLORS     = {"High": C_RED,     "Medium": C_BLUE,    "Low": C_GREEN}
CHURN_ICONS      = {"High": "🔴", "Medium": "🔵", "Low": "🟢"}

# Base Plotly layout applied to every chart
_LAYOUT = dict(
    margin=dict(l=10, r=10, t=44, b=10),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(size=12),
    xaxis=dict(tickangle=-30, automargin=True),
    yaxis=dict(automargin=True),
)


def _pct(v) -> str:
    return f"{v * 100:.1f}%" if pd.notna(v) else "—"


def _signed_pct(v) -> str:
    return f"{v * 100:+.1f}%" if pd.notna(v) else "—"


# ------------------------------------------------------------------ #
# Load data                                                            #
# ------------------------------------------------------------------ #
@st.cache_data(ttl=30)
def load_complaints() -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query("SELECT * FROM complaints ORDER BY created_at DESC", conn)
    conn.close()
    return df


df = load_complaints()

if df.empty:
    st.info("No complaints registered yet. Submit some via the chat to see analytics here.")
    st.stop()

df["created_at"]    = pd.to_datetime(df["created_at"], format="ISO8601")
df["risk_score"]    = pd.to_numeric(df["risk_score"],      errors="coerce")
df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
df["risk_pct"]      = df["risk_score"] * 100

st.button("🔄 Refresh", on_click=load_complaints.clear)

# ------------------------------------------------------------------ #
# KPIs                                                                 #
# ------------------------------------------------------------------ #
st.markdown("### Overview")
total     = len(df)
escalated = int(df["escalated"].sum())
resolved  = int((df["status"] == "Resolved").sum())
avg_risk  = df["risk_score"].mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Complaints", total)
k2.metric("Escalated",   escalated, delta=f"{escalated/total:.1%} of total", delta_color="inverse")
k3.metric("Resolved",    resolved,  delta=f"{resolved/total:.1%} of total")
k4.metric("Avg Risk Score", _pct(avg_risk))

st.divider()

# ------------------------------------------------------------------ #
# Distribution charts                                                  #
# ------------------------------------------------------------------ #
st.markdown("### Distribution")
c1, c2, c3 = st.columns(3)

with c1:
    cat_counts = df["category"].value_counts().sort_values().reset_index()
    cat_counts.columns = ["Category", "Count"]
    fig = px.bar(cat_counts, y="Category", x="Count", orientation="h",
                 title="Complaint Category",
                 color_discrete_sequence=[C_BLUE], text="Count")
    fig.update_traces(textposition="outside")
    fig.update_layout(**{**_LAYOUT, "xaxis": dict(automargin=True), "yaxis": dict(automargin=True)})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Volume of complaints per category. Highlights which product areas generate the most customer friction.")

with c2:
    pri_counts = (df["priority"].value_counts()
                  .reindex(["Critical", "High", "Medium", "Low"]).dropna().reset_index())
    pri_counts.columns = ["Priority", "Count"]
    fig = px.bar(pri_counts, x="Priority", y="Count", title="Priority Breakdown",
                 color="Priority", color_discrete_map=PRIORITY_COLORS, text="Count")
    fig.update_traces(textposition="outside")
    fig.update_layout(**_LAYOUT, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Priority is assigned by the risk scorer at registration time using 6 weighted factors: category severity, sentiment, unresolved count, complaint frequency, customer tier, and escalation history.")

with c3:
    sent_counts = df["sentiment"].value_counts().reset_index()
    sent_counts.columns = ["Sentiment", "Count"]
    fig = px.bar(sent_counts, x="Sentiment", y="Count", title="Sentiment Breakdown",
                 color="Sentiment", color_discrete_map=SENTIMENT_COLORS, text="Count")
    fig.update_traces(textposition="outside")
    fig.update_layout(**_LAYOUT, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("FinBERT sentiment label per complaint. Negative dominates as expected; positive entries indicate satisfied follow-up messages or resolved-case acknowledgements.")

st.divider()

# ------------------------------------------------------------------ #
# Risk Analysis                                                        #
# ------------------------------------------------------------------ #
st.markdown("### Risk Analysis")
c1, c2 = st.columns(2)

with c1:
    tier_order = ["Platinum", "Gold", "Standard"]
    conn = sqlite3.connect(str(DB_PATH))
    cust_df = pd.read_sql_query("SELECT customer_id, customer_value_tier FROM customers", conn)
    conn.close()
    merged = df.merge(cust_df, on="customer_id", how="left")
    risk_by_tier = (merged.groupby("customer_value_tier")["risk_pct"]
                    .mean().reindex(tier_order).dropna().reset_index())
    risk_by_tier.columns = ["Tier", "Avg Risk (%)"]
    risk_by_tier["Avg Risk (%)"] = risk_by_tier["Avg Risk (%)"].round(1)
    fig = px.bar(risk_by_tier, x="Tier", y="Avg Risk (%)",
                 title="Avg Risk Score by Customer Tier (%)",
                 color_discrete_sequence=[C_BLUE],
                 text=risk_by_tier["Avg Risk (%)"].apply(lambda v: f"{v:.1f}%"))
    fig.update_traces(textposition="outside")
    fig.update_layout(**_LAYOUT)
    fig.update_yaxes(ticksuffix="%", title="Avg Risk Score", range=[0, 105])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Higher-tier customers (Platinum/Gold) carry more weight in risk scoring, "
               "so their complaints tend to show elevated avg risk vs Standard tier.")

with c2:
    risk_by_cat = (df.groupby("category")["risk_pct"].mean()
                   .sort_values(ascending=True).reset_index())
    risk_by_cat.columns = ["Category", "Avg Risk (%)"]
    risk_by_cat["Avg Risk (%)"] = risk_by_cat["Avg Risk (%)"].round(1)
    use_horiz = len(risk_by_cat) > 7
    if use_horiz:
        fig = px.bar(risk_by_cat, y="Category", x="Avg Risk (%)", orientation="h",
                     title="Avg Risk Score by Category (%)",
                     color_discrete_sequence=[C_BLUE],
                     text=risk_by_cat["Avg Risk (%)"].apply(lambda v: f"{v:.1f}%"))
        fig.update_traces(textposition="outside")
        fig.update_layout(**{**_LAYOUT, "xaxis": dict(ticksuffix="%", automargin=True), "yaxis": dict(automargin=True)})
    else:
        fig = px.bar(risk_by_cat.sort_values("Avg Risk (%)", ascending=False),
                     x="Category", y="Avg Risk (%)",
                     title="Avg Risk Score by Category (%)",
                     color_discrete_sequence=[C_BLUE],
                     text=risk_by_cat["Avg Risk (%)"].apply(lambda v: f"{v:.1f}%"))
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(**_LAYOUT)
        fig.update_yaxes(ticksuffix="%", title="Avg Risk Score")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Average risk score per complaint category. Categories like Fraud and Unauthorized Transactions typically score higher due to their elevated base severity weight in the risk model.")

st.divider()

# ------------------------------------------------------------------ #
# Complaints over time                                                 #
# ------------------------------------------------------------------ #
st.markdown("### Complaints Over Time")

granularity = st.selectbox(
    "View by", options=["Hourly", "Daily", "Monthly"], index=1,
    key="timeline_granularity",
)

if granularity == "Hourly":
    df["_period"] = df["created_at"].dt.floor("h")
    x_label  = "Hour"
    tick_fmt = "%d %b, %H:%M"
elif granularity == "Monthly":
    df["_period"] = df["created_at"].dt.to_period("M").dt.to_timestamp()
    x_label  = "Month"
    tick_fmt = "%b %Y"
else:
    df["_period"] = df["created_at"].dt.floor("D")
    x_label  = "Date"
    tick_fmt = "%d %b %Y"

timeline = df.groupby("_period").size().reset_index(name="Complaints")
timeline.columns = [x_label, "Complaints"]

max_complaints = int(timeline["Complaints"].max()) if not timeline.empty else 5
y_max = max_complaints + 1

fig = px.line(timeline, x=x_label, y="Complaints", markers=True,
              title=f"{granularity} Complaint Volume",
              color_discrete_sequence=[C_BLUE])
fig.update_traces(marker=dict(size=7, color=C_BLUE), line=dict(width=2))
fig.update_layout(**_LAYOUT)
fig.update_xaxes(tickformat=tick_fmt, title=x_label)
fig.update_yaxes(
    title="Number of Complaints",
    range=[0, y_max],
    tickformat="d",
    nticks=min(y_max + 1, 8),
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Complaint volume over time grouped by the selected granularity. Spikes indicate peak complaint periods — useful for correlating with product changes, outages, or billing cycles.")

st.divider()

# ------------------------------------------------------------------ #
# Churn Analytics                                                      #
# ------------------------------------------------------------------ #
st.markdown("## 🔮 Customer Churn Analytics")
st.caption("Churn risk scored from complaint frequency, sentiment trend, escalation rate, unresolved rate and risk score history.")

@st.cache_data(ttl=30)
def load_churn():
    return compute_churn_report()

churn_report = load_churn()

if not churn_report.customers:
    st.info("No customer complaint data available yet for churn analysis.")
else:
    ck1, ck2, ck3, ck4 = st.columns(4)
    ck1.metric("Customers Analysed", len(churn_report.customers))
    ck2.metric("High Risk 🔴", churn_report.high_risk_count,
               delta=f"{churn_report.high_risk_count/len(churn_report.customers):.1%}",
               delta_color="inverse")
    ck3.metric("Medium Risk 🔵", churn_report.medium_risk_count)
    ck4.metric("Avg Churn Score", _pct(churn_report.avg_churn_score))

    st.divider()

    churn_df = pd.DataFrame([c.to_dict() for c in churn_report.customers])

    col1, col2 = st.columns(2)
    with col1:
        risk_dist = (churn_df["churn_risk"].value_counts()
                     .reindex(["High", "Medium", "Low"]).dropna().reset_index())
        risk_dist.columns = ["Churn Risk", "Count"]
        fig = px.bar(risk_dist, x="Churn Risk", y="Count", title="Churn Risk Distribution",
                     color="Churn Risk", color_discrete_map=CHURN_COLORS, text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(**_LAYOUT, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Number of customers in each churn risk tier. High (≥65%) requires immediate retention action; Medium (50–65%) warrants proactive outreach; Low (<50%) is stable. Thresholds are calibrated to the dataset's score distribution.")

    with col2:
        score_df = churn_df.sort_values("churn_score", ascending=False).copy()
        score_df["churn_pct"] = (score_df["churn_score"] * 100).round(1)
        fig = px.bar(score_df, x="customer_id", y="churn_pct",
                     title="Churn Score by Customer (%)",
                     color="churn_risk", color_discrete_map=CHURN_COLORS,
                     text=score_df["churn_pct"].apply(lambda v: f"{v:.1f}%"),
                     labels={"customer_id": "Customer ID", "churn_pct": "Churn Score (%)", "churn_risk": "Risk"})
        fig.update_traces(textposition="outside")
        fig.update_layout(**_LAYOUT)
        fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Individual churn scores ranked highest to lowest. Color reflects risk tier — red bars need urgent attention. Score is a weighted composite of 10 behavioral and complaint-history features.")

    st.divider()

    st.markdown("### Customer Churn Risk Table")
    st.caption("Full churn feature matrix for all customers. Each row shows the 10 input signals used to compute the churn score — useful for understanding why a customer was assigned a particular risk tier.")
    churn_display = churn_df[[
        "customer_id", "churn_risk", "churn_score",
        "total_complaints", "complaint_frequency_30d",
        "avg_sentiment_polarity", "sentiment_trend",
        "escalation_rate", "unresolved_rate",
        "avg_risk_score", "max_risk_score",
        "critical_count", "days_since_last_complaint",
    ]].copy()

    churn_display["churn_risk"]             = churn_display["churn_risk"].map(lambda r: f"{CHURN_ICONS.get(r,'')} {r}")
    churn_display["churn_score"]            = churn_display["churn_score"].apply(_pct)
    churn_display["avg_sentiment_polarity"] = churn_display["avg_sentiment_polarity"].apply(_signed_pct)
    churn_display["sentiment_trend"]        = churn_display["sentiment_trend"].apply(lambda v: f"{v:+.4f}")
    churn_display["escalation_rate"]        = churn_display["escalation_rate"].apply(_pct)
    churn_display["unresolved_rate"]        = churn_display["unresolved_rate"].apply(_pct)
    churn_display["avg_risk_score"]         = churn_display["avg_risk_score"].apply(_pct)
    churn_display["max_risk_score"]         = churn_display["max_risk_score"].apply(_pct)

    churn_display = churn_display.rename(columns={
        "customer_id":               "Customer ID",
        "churn_risk":                "Churn Risk",
        "churn_score":               "Churn Score",
        "total_complaints":          "Total Complaints",
        "complaint_frequency_30d":   "Complaints (30d)",
        "avg_sentiment_polarity":    "Avg Sentiment",
        "sentiment_trend":           "Sentiment Trend",
        "escalation_rate":           "Escalation Rate",
        "unresolved_rate":           "Unresolved Rate",
        "avg_risk_score":            "Avg Risk Score",
        "max_risk_score":            "Max Risk Score",
        "critical_count":            "Critical Complaints",
        "days_since_last_complaint": "Days Since Last",
    })
    st.dataframe(churn_display, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("### Churn Feature Breakdown")
    st.caption("Drill into an individual customer's churn drivers. Select a customer to see each feature value alongside its interpretation — helpful for tailoring retention conversations.")
    selected_cid = st.selectbox(
        "Select a customer to inspect churn drivers",
        options=[c.customer_id for c in churn_report.customers],
    )
    if selected_cid:
        c = next(x for x in churn_report.customers if x.customer_id == selected_cid)
        st.markdown(
            f"**{c.customer_id}** — Churn Score: `{_pct(c.churn_score)}` "
            f"&nbsp; {CHURN_ICONS.get(c.churn_risk,'')} **{c.churn_risk} Risk**"
        )
        feat_data = {
            "Feature": [
                "Avg Sentiment Polarity", "Sentiment Trend",
                "Escalation Rate", "Unresolved Rate",
                "Avg Risk Score", "Max Risk Score",
                "Complaints (30d)", "Complaints (90d)",
                "Critical Complaints", "Days Since Last Complaint",
            ],
            "Value": [
                _signed_pct(c.avg_sentiment_polarity), f"{c.sentiment_trend:+.4f}",
                _pct(c.escalation_rate), _pct(c.unresolved_rate),
                _pct(c.avg_risk_score), _pct(c.max_risk_score),
                c.complaint_frequency_30d, c.complaint_frequency_90d,
                c.critical_count, c.days_since_last_complaint,
            ],
            "Interpretation": [
                "More negative → higher churn risk",
                "Negative slope → worsening sentiment",
                "Higher escalation → lower trust",
                "Unresolved complaints drive dissatisfaction",
                "Higher severity complaints",
                "Worst single complaint experienced",
                "Recent complaint surge",
                "Sustained complaint volume",
                "Severe unresolved issues",
                "Lower = more recently active/frustrated",
            ],
        }
        st.dataframe(pd.DataFrame(feat_data), use_container_width=True, hide_index=True)

    # Feature importance + model comparison (shown when model exists)
    _churn_model_path = "models/churn_classifier.pkl"
    import os, joblib
    if os.path.exists(_churn_model_path):
        st.divider()
        _bundle      = joblib.load(_churn_model_path)
        _model_name  = _bundle.get("model_name", "RandomForest")
        _loocv_res   = _bundle.get("loocv_results", {})
        _importances = _bundle["model"].feature_importances_
        _feat_names  = _bundle["feature_names"]
        from src.analytics.churn_analyzer import _WEIGHTS as _CHURN_WEIGHTS

        st.markdown(f"### Churn Model Comparison & Feature Importance")
        st.caption(
            f"Random Forest vs Gradient Boosting evaluated via Leave-One-Out CV (n={len(churn_report.customers)}). "
            f"Active model: **{_model_name}** (best LOOCV accuracy). "
            "Feature importances compared against hand-coded rule weights to validate domain assumptions."
        )

        # LOOCV comparison bar chart
        if _loocv_res:
            _loocv_df = pd.DataFrame([
                {"Model": k, "LOOCV Accuracy (%)": round(v["accuracy"] * 100, 1)}
                for k, v in _loocv_res.items()
            ])
            _active_color = [C_BLUE if r["Model"] == _model_name else C_GREY
                             for _, r in _loocv_df.iterrows()]
            fig_loocv = px.bar(
                _loocv_df, x="Model", y="LOOCV Accuracy (%)",
                title="Churn Model LOOCV Comparison",
                color_discrete_sequence=[C_BLUE],
                text=_loocv_df["LOOCV Accuracy (%)"].apply(lambda v: f"{v:.1f}%"),
            )
            fig_loocv.update_traces(textposition="outside",
                                    marker_color=_active_color)
            fig_loocv.update_layout(**{**_LAYOUT, "yaxis": dict(range=[0, 115], ticksuffix="%", automargin=True)})
            st.plotly_chart(fig_loocv, use_container_width=True)

        _fi_df = pd.DataFrame({
            "Feature":    _feat_names,
            "ML Importance": (_importances * 100).round(2),
            "Rule Weight":   [_CHURN_WEIGHTS.get(n, 0.0) * 100 for n in _feat_names],
        }).sort_values("ML Importance", ascending=False)

        _base = {k: v for k, v in _LAYOUT.items() if k not in ("xaxis", "yaxis")}

        fig_fi = px.bar(
            _fi_df.sort_values("ML Importance", ascending=True),
            x="ML Importance", y="Feature", orientation="h",
            title=f"{_model_name} — Feature Importances (%)",
            color_discrete_sequence=[C_BLUE],
            text=_fi_df.sort_values("ML Importance", ascending=True)["ML Importance"].apply(lambda v: f"{v:.1f}%"),
        )
        fig_fi.update_traces(textposition="outside")
        fig_fi.update_layout(**_base,
                             xaxis=dict(range=[0, _fi_df["ML Importance"].max() * 1.3],
                                        ticksuffix="%", automargin=True),
                             yaxis=dict(automargin=True))

        fig_rw = px.bar(
            _fi_df.sort_values("Rule Weight", ascending=True),
            x="Rule Weight", y="Feature", orientation="h",
            title="Hand-coded Rule Weights (%)",
            color_discrete_sequence=[C_ORANGE],
            text=_fi_df.sort_values("Rule Weight", ascending=True)["Rule Weight"].apply(lambda v: f"{v:.1f}%"),
        )
        fig_rw.update_traces(textposition="outside")
        fig_rw.update_layout(**_base,
                             xaxis=dict(range=[0, _fi_df["Rule Weight"].max() * 1.3],
                                        ticksuffix="%", automargin=True),
                             yaxis=dict(automargin=True))

        fi_col1, fi_col2 = st.columns(2)
        with fi_col1:
            st.plotly_chart(fig_fi, use_container_width=True)
        with fi_col2:
            st.plotly_chart(fig_rw, use_container_width=True)

        st.dataframe(
            _fi_df.rename(columns={"ML Importance": "ML Importance (%)", "Rule Weight": "Rule Weight (%)"}),
            use_container_width=True, hide_index=True,
        )

st.divider()

# ------------------------------------------------------------------ #
# XAI Insights                                                         #
# ------------------------------------------------------------------ #
st.markdown("## 🔎 AI Transparency (XAI)")
st.caption(
    "Explainability tools for the category classifier and risk scorer — "
    "showing what the models have learned and why individual decisions were made."
)

xai_tab1, xai_tab2 = st.tabs(["Category Classifier", "Risk Factor Breakdown"])

with xai_tab1:
    st.markdown("### Top Predictive Terms per Category")
    st.caption(
        "Highest-weight word n-grams from the Logistic Regression coefficients. "
        "These are the terms that most strongly associate a complaint with a given category."
    )
    from src.classification.complaint_classifier import get_class_top_terms, get_top_features

    _cat_labels_xai = [
        "Fraud & Unauthorized Charges", "Billing Concern", "Transaction Dispute",
        "Account Access Issue", "Card Services", "Reward & Points Issue",
        "Loan & Credit Issue", "Customer Service Complaint", "General Inquiry",
    ]
    _sel = st.selectbox("Category", options=_cat_labels_xai, key="xai_analytics_cat")
    _terms = get_class_top_terms(_sel, n=12)
    if _terms:
        _tdf = pd.DataFrame(_terms, columns=["Term", "LR Weight"]).sort_values("LR Weight")
        fig_xai = px.bar(_tdf, x="LR Weight", y="Term", orientation="h",
                         title=f"Top Terms → {_sel}",
                         color_discrete_sequence=[C_BLUE],
                         text=_tdf["LR Weight"].apply(lambda v: f"{v:.3f}"))
        fig_xai.update_traces(textposition="outside")
        fig_xai.update_layout(
            **{**_LAYOUT, "xaxis": dict(automargin=True), "yaxis": dict(automargin=True)}
        )
        st.plotly_chart(fig_xai, use_container_width=True)

        st.markdown("---")
        st.markdown("### Explain a Complaint Classification")
        _xai_text = st.text_area(
            "Enter complaint text:", key="xai_analytics_input", height=80,
            placeholder="e.g. My credit card was used for a transaction I never made…"
        )
        if st.button("Explain", key="xai_analytics_btn") and _xai_text.strip():
            _pred, _contribs = get_top_features(_xai_text.strip(), n=8)
            if _pred:
                st.success(f"Predicted category: **{_pred}**")
                if _contribs:
                    _cdf = pd.DataFrame(_contribs, columns=["Term", "Contribution"]).sort_values("Contribution")
                    fig_c2 = px.bar(_cdf, x="Contribution", y="Term", orientation="h",
                                    title="Why was this complaint classified here?",
                                    color_discrete_sequence=[C_BLUE],
                                    text=_cdf["Contribution"].apply(lambda v: f"{v:.3f}"))
                    fig_c2.update_traces(textposition="outside")
                    fig_c2.update_layout(
                        **{**_LAYOUT, "xaxis": dict(automargin=True), "yaxis": dict(automargin=True)}
                    )
                    st.plotly_chart(fig_c2, use_container_width=True)
                else:
                    st.info("No strong word-level signals found for this text.")
            else:
                st.warning("Could not generate explanation. Check that the classifier model is loaded correctly.")
    else:
        st.info("Category classifier not trained yet. Run `python scripts/train_category_classifier.py`.")

with xai_tab2:
    st.markdown("### Risk Factor Contribution — Individual Complaint")
    st.caption(
        "Select any complaint to see how each of the 6 risk factors contributed to its total score. "
        "Bars show the weighted contribution of each factor to the final risk score."
    )
    _complaint_ids = df["complaint_id"].tolist() if not df.empty else []
    if _complaint_ids:
        _sel_cid = st.selectbox("Select complaint", options=_complaint_ids, key="xai_risk_sel")
        _row = df[df["complaint_id"] == _sel_cid].iloc[0]

        from src.classification.risk_scorer import compute_risk_score
        import sqlite3 as _sq3
        _conn2 = _sq3.connect(str(DB_PATH))
        _cust_tier_row = _conn2.execute(
            "SELECT customer_value_tier FROM customers WHERE customer_id=?",
            (_row["customer_id"],)
        ).fetchone()
        _conn2.close()
        _tier = _cust_tier_row[0] if _cust_tier_row else "Standard"

        _rs = compute_risk_score(
            category=str(_row.get("category", "General Inquiry")),
            complaint_text=str(_row.get("anonymized_text", "")),
            sentiment_polarity=float(_row.get("sentiment_score", 0.0) or 0.0),
            escalation_count=int(_row.get("escalated", 0) or 0),
            unresolved_count=0,
            complaint_frequency_30d=1,
            customer_value_tier=_tier,
        )
        _bd = _rs.breakdown()

        _factors_df = pd.DataFrame([
            {"Factor": "Severity",        "Contribution": _bd["severity"]},
            {"Factor": "Sentiment",       "Contribution": _bd["sentiment"]},
            {"Factor": "Escalation",      "Contribution": _bd["escalation"]},
            {"Factor": "Unresolved",      "Contribution": _bd["unresolved"]},
            {"Factor": "Frequency",       "Contribution": _bd["frequency"]},
            {"Factor": "Customer Value",  "Contribution": _bd["customer_value"]},
        ]).sort_values("Contribution", ascending=True)

        fig_bd = px.bar(
            _factors_df, x="Contribution", y="Factor", orientation="h",
            title=f"Risk Factor Breakdown — {_sel_cid}  (Total: {_bd['total_risk_score']:.2f} → {_bd['priority']})",
            color_discrete_sequence=[C_BLUE],
            text=_factors_df["Contribution"].apply(lambda v: f"{v:.3f}"),
        )
        fig_bd.update_traces(textposition="outside")
        fig_bd.update_layout(
            **{**_LAYOUT, "xaxis": dict(range=[0, 1.2], automargin=True),
               "yaxis": dict(automargin=True)}
        )
        st.plotly_chart(fig_bd, use_container_width=True)

        st.markdown(f"""
| Factor | Raw Value | Risk Contribution |
|---|---|---|
| **Severity** (category base + keyword boosts) | {_row.get('category', '—')} | `{_bd['severity']:.3f}` |
| **Sentiment** (1 − polarity)/2 | {_row.get('sentiment_score', '—')} | `{_bd['sentiment']:.3f}` |
| **Escalation** | {'Yes' if _row.get('escalated') else 'No'} | `{_bd['escalation']:.3f}` |
| **Customer Value** tier multiplier | {_tier} | `{_bd['customer_value']:.3f}` |
| **Total Risk Score** | | `{_bd['total_risk_score']:.4f}` |
| **Priority** | | **{_bd['priority']}** |
""")
    else:
        st.info("No complaint data available. Register some complaints first.")

st.divider()

# ------------------------------------------------------------------ #
# Complaint Records table                                              #
# ------------------------------------------------------------------ #
st.markdown("### Complaint Records")
st.caption("Filterable log of all registered complaints. Use the dropdowns to slice by category, priority, or sentiment — the selection also drives the Complaint Detail Inspector below.")

f1, f2, f3 = st.columns(3)
cat_filter  = f1.multiselect("Category", options=sorted(df["category"].dropna().unique()))
pri_filter  = f2.multiselect("Priority", options=["Critical", "High", "Medium", "Low"])
sent_filter = f3.multiselect("Sentiment", options=sorted(df["sentiment"].dropna().unique()))

filtered = df.copy()
if cat_filter:  filtered = filtered[filtered["category"].isin(cat_filter)]
if pri_filter:  filtered = filtered[filtered["priority"].isin(pri_filter)]
if sent_filter: filtered = filtered[filtered["sentiment"].isin(sent_filter)]

display_cols = [
    "complaint_id", "customer_id", "category", "priority",
    "risk_score", "sentiment", "sentiment_score", "status",
    "escalated", "resolution_attempts", "created_at",
]
display_df = filtered[display_cols].copy()
display_df["risk_score"]      = display_df["risk_score"].apply(_pct)
display_df["sentiment_score"] = display_df["sentiment_score"].apply(_signed_pct)

st.dataframe(
    display_df.rename(columns={
        "complaint_id":        "Complaint ID",
        "customer_id":         "Customer ID",
        "category":            "Category",
        "priority":            "Priority",
        "risk_score":          "Risk Score",
        "sentiment":           "Sentiment",
        "sentiment_score":     "Sentiment Polarity",
        "status":              "Status",
        "escalated":           "Escalated",
        "resolution_attempts": "Resolution Attempts",
        "created_at":          "Registered At",
    }),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ------------------------------------------------------------------ #
# Complaint Detail Inspector — pinned to bottom                       #
# ------------------------------------------------------------------ #
st.markdown("### Complaint Detail Inspector")
st.caption("Deep-dive into a single complaint — view the anonymized text, the agent's mitigation response, NLP outputs (sentiment, intent, entities), risk score breakdown, and lifecycle status.")
complaint_ids = filtered["complaint_id"].tolist()
selected_id = st.selectbox("Select a complaint to inspect", options=complaint_ids)

if selected_id:
    row = df[df["complaint_id"] == selected_id].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Complaint Text (Anonymized)**")
        st.info(row["anonymized_text"] or "—")
        st.markdown("**Mitigation Response**")
        st.success(row["mitigation_response"] or "—")

    with col2:
        st.markdown("**Classification**")
        m1, m2 = st.columns(2)
        m1.metric("Category", row["category"] or "—")
        m2.metric("Intent", str(row["intent"] or "—").replace("_", " ").title())

        st.markdown("**Sentiment**")
        m1, m2 = st.columns(2)
        m1.metric("Label", str(row["sentiment"] or "—").title())
        m2.metric("Polarity", _signed_pct(row["sentiment_score"]))

        st.markdown("**Risk Scoring**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Score", _pct(row["risk_score"]))
        m2.metric("Priority", row["priority"] or "—")
        m3.metric("Escalated", "Yes" if row["escalated"] else "No")

        st.markdown("**Lifecycle**")
        m1, m2 = st.columns(2)
        m1.metric("Status", row["status"] or "—")
        m2.metric("Resolution Attempts", int(row["resolution_attempts"] or 0))

        st.markdown("**Customer ID**")
        st.code(row["customer_id"])

        if row["entities"]:
            st.markdown("**Extracted Entities**")
            try:
                entities = json.loads(row["entities"]) if isinstance(row["entities"], str) else row["entities"]
                st.json(entities)
            except Exception:
                st.write(row["entities"])
