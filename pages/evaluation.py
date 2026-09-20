"""
System Evaluation Dashboard — metrics for the dissertation.
Accessible to admin only.
"""
import json
import os
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from src.classification.complaint_classifier import get_class_top_terms, get_top_features

if not st.session_state.get("authenticated") or st.session_state.customer.role != "admin":
    st.error("Access denied.")
    st.stop()

RESULTS_FILE = "data/eval/results.json"

C_BLUE   = "#2980B9"
C_GREEN  = "#27AE60"
C_ORANGE = "#E67E22"
C_RED    = "#C0392B"

_LAYOUT = dict(
    margin=dict(l=10, r=10, t=44, b=10),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(size=12),
)

st.title("🧪 System Evaluation")
st.caption("Classification accuracy, precision, recall, F1-score and retrieval relevance metrics for the dissertation prototype.")

with st.sidebar:
    st.success(f"Logged in as **{st.session_state.customer.name}**")
    st.caption("Role: Admin")
    if st.button("Logout", use_container_width=True):
        for key in ["authenticated", "customer"]:
            st.session_state.pop(key, None)
        st.rerun()

if not os.path.exists(RESULTS_FILE):
    st.warning("No evaluation results found. Run `python scripts/evaluate_system.py` first.")
    st.stop()

with open(RESULTS_FILE) as f:
    R = json.load(f)


# ------------------------------------------------------------------ #
# Top-level KPIs                                                       #
# ------------------------------------------------------------------ #
st.markdown("### Overall Performance")
st.caption("Summary of accuracy and macro-averaged F1 across all evaluated components.")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Intent Detection",    f"{R['intent_detection']['accuracy']*100:.1f}%",
          delta=f"F1 {R['intent_detection']['report']['macro avg']['f1-score']*100:.1f}%")
_cat_best = R.get("category_classification_trained") or R["category_classification"]
k2.metric("Category Classif.",   f"{_cat_best['accuracy']*100:.1f}%",
          delta=f"F1 {_cat_best['report']['macro avg']['f1-score']*100:.1f}%")
k3.metric("Sentiment Analysis",  f"{R['sentiment_analysis']['accuracy']*100:.1f}%",
          delta=f"F1 {R['sentiment_analysis']['report']['macro avg']['f1-score']*100:.1f}%")
k4.metric("RAG Precision@1",     f"{R['rag_retrieval']['precision_at_1']*100:.1f}%",
          delta=f"P@3 {R['rag_retrieval']['precision_at_3']*100:.1f}%")
k5.metric("Risk Score Accuracy", f"{R['risk_scoring']['accuracy']*100:.1f}%")

st.divider()

# ------------------------------------------------------------------ #
# Component deep-dives                                                 #
# ------------------------------------------------------------------ #
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Intent Detection",
    "🗂️ Category Classification",
    "💬 Sentiment Analysis",
    "📚 RAG Retrieval",
    "⚖️ Risk Scoring",
])


def _per_class_df(report, labels):
    rows = []
    for lbl in labels:
        if lbl in report:
            r = report[lbl]
            rows.append({
                "Class": lbl,
                "Precision": round(r["precision"], 3),
                "Recall":    round(r["recall"], 3),
                "F1-Score":  round(r["f1-score"], 3),
                "Support":   int(r["support"]),
            })
    return pd.DataFrame(rows)


def _confusion_heatmap(cm, labels, title):
    fig = ff.create_annotated_heatmap(
        z=cm, x=labels, y=labels,
        colorscale="Blues", showscale=True,
        annotation_text=[[str(v) for v in row] for row in cm],
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        **{k: v for k, v in _LAYOUT.items() if k not in ("xaxis", "yaxis")},
        xaxis=dict(tickangle=-30, automargin=True),
        yaxis=dict(automargin=True, autorange="reversed"),
    )
    return fig


# ---- Intent Detection ------------------------------------------------
with tab1:
    st.markdown("### Intent Detection")
    st.caption(
        "Pattern-based + zero-shot BART-MNLI intent classifier evaluated on 50 hand-labelled samples "
        "covering all 7 intents: register_complaint, track_complaint, close_complaint, "
        "escalate_to_agent, resolve_query, greet, goodbye."
    )
    intent_labels = ["register_complaint", "track_complaint", "close_complaint",
                     "escalate_to_agent", "resolve_query", "greet", "goodbye"]

    col1, col2 = st.columns([1, 1])
    with col1:
        df_i = _per_class_df(R["intent_detection"]["report"], intent_labels)
        st.dataframe(df_i, use_container_width=True, hide_index=True)

        macro = R["intent_detection"]["report"]["macro avg"]
        st.markdown(f"""
| Metric | Value |
|---|---|
| **Accuracy** | {R['intent_detection']['accuracy']*100:.1f}% |
| **Macro Precision** | {macro['precision']*100:.1f}% |
| **Macro Recall** | {macro['recall']*100:.1f}% |
| **Macro F1** | {macro['f1-score']*100:.1f}% |
""")
    with col2:
        fig = _confusion_heatmap(
            R["intent_detection"]["confusion_matrix"],
            [l.replace("_", " ") for l in intent_labels],
            "Intent Confusion Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Observations**")
    st.info(
        "Intent detection achieves **88% accuracy** with strong performance on complaint registration, "
        "tracking, closure, and escalation intents. The main misclassification occurs for general "
        "queries (resolve_query) where phrases like 'How can I increase my credit limit?' overlap "
        "lexically with complaint patterns. This is expected behaviour as the pattern-based layer "
        "prioritises complaint registration to minimise missed complaints."
    )

# ---- Category Classification -----------------------------------------
with tab2:
    cat_labels = [
        "Fraud & Unauthorized Charges", "Billing Concern", "Transaction Dispute",
        "Account Access Issue", "Card Services", "Reward & Points Issue",
        "Loan & Credit Issue", "Customer Service Complaint", "General Inquiry",
    ]
    short_labels = [
        l.replace(" & ", "/").replace(" Issue", "").replace(" Concern", "")
         .replace(" Charges", "").replace(" Complaint", "")
         .replace("Unauthorized", "Unauth.") for l in cat_labels
    ]

    trained    = R.get("category_classification_trained")
    distilbert = R.get("category_classification_distilbert")

    zs_acc = R["category_classification"]["accuracy"]
    zs_f1  = R["category_classification"]["report"]["macro avg"]["f1-score"]
    tr_acc = trained["accuracy"]                               if trained    else None
    tr_f1  = trained["report"]["macro avg"]["f1-score"]        if trained    else None
    db_acc = distilbert["accuracy"]                            if distilbert else None
    db_f1  = distilbert["report"]["macro avg"]["f1-score"]     if distilbert else None

    st.markdown("### Complaint Category Classification")
    st.caption(
        "Three-tier pipeline: keyword patterns (Tier 1) → TF-IDF + LR (Tier 2, active) → "
        "BART-MNLI zero-shot (Tier 3, fallback). Tier 1 is embedded in the live pipeline and not "
        "independently benchmarked here. Sections below compare the two evaluable methods "
        "(zero-shot vs trained) plus DistilBERT fine-tuned on the same data for dissertation comparison. "
        "Test set: 45 hand-labelled samples, 5 per class across 9 categories."
    )

    # ────────────────────────────────────────────────────────────────────
    # A. Model Comparison Summary
    # ────────────────────────────────────────────────────────────────────
    st.markdown("#### Model Comparison Summary")

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Zero-shot  (BART-MNLI)", f"{zs_acc*100:.1f}%",
                delta=f"F1 {zs_f1*100:.1f}%", delta_color="off")
    if tr_acc is not None:
        kpi2.metric("TF-IDF + LR  ✅  Active", f"{tr_acc*100:.1f}%",
                    delta=f"{(tr_acc - zs_acc)*100:+.1f}% vs zero-shot")
    if db_acc is not None:
        kpi3.metric("DistilBERT  (eval only)", f"{db_acc*100:.1f}%",
                    delta=f"{(db_acc - zs_acc)*100:+.1f}% vs zero-shot")

    model_order = [("Zero-shot\n(BART-MNLI)", zs_acc, zs_f1, C_BLUE, "#85C1E9")]
    if tr_acc is not None:
        model_order.append(("TF-IDF + LR\n(Active)", tr_acc, tr_f1, C_BLUE, "#85C1E9"))
    if db_acc is not None:
        model_order.append(("DistilBERT\n(Eval only)", db_acc, db_f1, C_BLUE, "#85C1E9"))

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(
        name="Accuracy",
        x=[m[0] for m in model_order],
        y=[m[1] * 100 for m in model_order],
        marker_color=[m[3] for m in model_order],
        text=[f"{m[1]*100:.1f}%" for m in model_order],
        textposition="outside", offsetgroup=0,
    ))
    fig_cmp.add_trace(go.Bar(
        name="Macro F1",
        x=[m[0] for m in model_order],
        y=[m[2] * 100 for m in model_order],
        marker_color=[m[4] for m in model_order],
        text=[f"{m[2]*100:.1f}%" for m in model_order],
        textposition="outside", offsetgroup=1,
    ))
    fig_cmp.update_layout(
        **{**_LAYOUT, "margin": dict(l=10, r=10, t=44, b=60)},
        title="Accuracy & Macro F1 by Classification Method",
        barmode="group",
        yaxis=dict(range=[0, 115], ticksuffix="%", automargin=True),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.divider()

    # ────────────────────────────────────────────────────────────────────
    # B. Tier 3 — BART-MNLI Zero-shot (Fallback Baseline)
    # ────────────────────────────────────────────────────────────────────
    st.markdown("#### BART-MNLI Zero-shot  _(Fallback / Dissertation Baseline)_")
    st.caption(
        "Pure zero-shot classification via facebook/bart-large-mnli — no training data, no patterns. "
        "Active only when both Tier 1 and Tier 2 yield no result. Also used as the unbiased baseline "
        "for comparing all three methods."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        df_zs = _per_class_df(R["category_classification"]["report"], cat_labels)
        st.dataframe(df_zs, use_container_width=True, hide_index=True)
        macro_zs = R["category_classification"]["report"]["macro avg"]
        st.markdown(f"""
| Metric | Value |
|---|---|
| **Accuracy** | {zs_acc*100:.1f}% |
| **Macro Precision** | {macro_zs['precision']*100:.1f}% |
| **Macro Recall** | {macro_zs['recall']*100:.1f}% |
| **Macro F1** | {macro_zs['f1-score']*100:.1f}% |
""")
        df_zs_sorted = df_zs.sort_values("F1-Score", ascending=True)
        df_zs_sorted["F1 (%)"] = (df_zs_sorted["F1-Score"] * 100).round(1)
        fig_zs = px.bar(df_zs_sorted, x="F1 (%)", y="Class", orientation="h",
                        title="Zero-shot — F1 per Category",
                        color_discrete_sequence=[C_BLUE],
                        text=df_zs_sorted["F1 (%)"].apply(lambda v: f"{v:.1f}%"))
        fig_zs.update_traces(textposition="outside")
        fig_zs.update_layout(**_LAYOUT,
                             xaxis=dict(range=[0, 115], ticksuffix="%", automargin=True),
                             yaxis=dict(automargin=True))
        st.plotly_chart(fig_zs, use_container_width=True)
    with col2:
        st.plotly_chart(
            _confusion_heatmap(R["category_classification"]["confusion_matrix"],
                               short_labels, "Zero-shot Confusion Matrix"),
            use_container_width=True,
        )

    st.info(
        f"Zero-shot BART-MNLI achieves **{zs_acc*100:.1f}% accuracy** with no training. "
        "Key confusion clusters: Card Services ↔ Account Access Issue (overlapping card/account vocabulary), "
        "Loan & Credit ↔ Billing Concern (both involve charges), and General Inquiry misclassified "
        "as Account Access (query phrasing overlap)."
    )

    # ────────────────────────────────────────────────────────────────────
    # C. Tier 2 — TF-IDF + Logistic Regression (Active)
    # ────────────────────────────────────────────────────────────────────
    if trained:
        st.divider()
        st.markdown("#### TF-IDF + Logistic Regression  ✅  Active in Live Pipeline")
        st.caption(
            "FeatureUnion: word n-grams (1–2, 10 k features) + char n-grams (3–5 char_wb, 8 k features), "
            "sublinear TF scaling. Logistic Regression with balanced class weights. "
            "Trained on 701 samples (566 from SQLite + 135 hand-crafted supplementary examples). "
            "This is the active Tier 2 classifier; BART-MNLI zero-shot is the Tier 3 fallback."
        )

        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            df_t = _per_class_df(trained["report"], cat_labels)
            st.dataframe(df_t, use_container_width=True, hide_index=True)
            macro_t = trained["report"]["macro avg"]
            st.markdown(f"""
| Metric | Value |
|---|---|
| **Accuracy** | {tr_acc*100:.1f}% |
| **Macro Precision** | {macro_t['precision']*100:.1f}% |
| **Macro Recall** | {macro_t['recall']*100:.1f}% |
| **Macro F1** | {macro_t['f1-score']*100:.1f}% |
""")
            df_t_sorted = df_t.sort_values("F1-Score", ascending=True)
            df_t_sorted["F1 (%)"] = (df_t_sorted["F1-Score"] * 100).round(1)
            fig_t = px.bar(df_t_sorted, x="F1 (%)", y="Class", orientation="h",
                           title="TF-IDF + LR — F1 per Category",
                           color_discrete_sequence=[C_BLUE],
                           text=df_t_sorted["F1 (%)"].apply(lambda v: f"{v:.1f}%"))
            fig_t.update_traces(textposition="outside")
            fig_t.update_layout(**_LAYOUT,
                                xaxis=dict(range=[0, 115], ticksuffix="%", automargin=True),
                                yaxis=dict(automargin=True))
            st.plotly_chart(fig_t, use_container_width=True)
        with col_t2:
            st.plotly_chart(
                _confusion_heatmap(trained["confusion_matrix"],
                                   short_labels, "TF-IDF + LR Confusion Matrix"),
                use_container_width=True,
            )

        st.success(
            f"TF-IDF + LR achieves **{tr_acc*100:.1f}% accuracy** — a "
            f"**{(tr_acc - zs_acc)*100:+.1f}%** improvement over the {zs_acc*100:.1f}% zero-shot baseline. "
            "Training on in-domain financial complaint vocabulary teaches class-specific n-gram patterns "
            "that zero-shot BART-MNLI cannot capture without domain exposure. "
            "Character n-grams improve recall for short or misspelled complaint phrases."
        )

        # ── XAI: Classifier Interpretability ─────────────────────────────
        with st.expander("🔍 Classifier Interpretability (XAI)"):
            st.markdown("#### Top Predictive Terms per Category")
            st.caption(
                "Highest-weight word n-grams from the Logistic Regression coefficients. "
                "These are the terms that most strongly associate a complaint with a given category — "
                "derived analytically from the trained model weights, not from any specific input."
            )
            xai_col1, xai_col2 = st.columns([1, 2])
            with xai_col1:
                sel_cat = st.selectbox(
                    "Select category", options=cat_labels, key="xai_cat_sel"
                )
            with xai_col2:
                terms = get_class_top_terms(sel_cat, n=12)
                if terms:
                    terms_df = pd.DataFrame(terms, columns=["Term", "LR Weight"])
                    terms_df = terms_df.sort_values("LR Weight")
                    fig_terms = px.bar(
                        terms_df, x="LR Weight", y="Term", orientation="h",
                        title=f"Top Terms → {sel_cat}",
                        color_discrete_sequence=[C_BLUE],
                        text=terms_df["LR Weight"].apply(lambda v: f"{v:.3f}"),
                    )
                    fig_terms.update_traces(textposition="outside")
                    fig_terms.update_layout(**_LAYOUT,
                                           xaxis=dict(automargin=True),
                                           yaxis=dict(automargin=True))
                    st.plotly_chart(fig_terms, use_container_width=True)
                else:
                    st.info("Run `python scripts/train_category_classifier.py` first.")

            st.markdown("---")
            st.markdown("#### Explain a Single Prediction")
            st.caption(
                "Enter any complaint text to see which words pushed the model toward its predicted category. "
                "Contribution = TF-IDF weight × LR coefficient for the predicted class."
            )
            explain_text = st.text_area(
                "Complaint text:", key="xai_explain_input", height=80,
                placeholder="e.g. I was charged twice for the same transaction last week…"
            )
            if st.button("Explain Classification", key="xai_explain_btn") and explain_text.strip():
                pred_label, contribs = get_top_features(explain_text.strip(), n=8)
                if pred_label:
                    st.success(f"Predicted category: **{pred_label}**")
                    if contribs:
                        contrib_df = pd.DataFrame(contribs, columns=["Term", "Contribution"])
                        contrib_df = contrib_df.sort_values("Contribution")
                        fig_c = px.bar(
                            contrib_df, x="Contribution", y="Term", orientation="h",
                            title="Top contributing terms for this prediction",
                            color_discrete_sequence=[C_BLUE],
                            text=contrib_df["Contribution"].apply(lambda v: f"{v:.3f}"),
                        )
                        fig_c.update_traces(textposition="outside")
                        fig_c.update_layout(**_LAYOUT,
                                           xaxis=dict(automargin=True),
                                           yaxis=dict(automargin=True))
                        st.plotly_chart(fig_c, use_container_width=True)
                    else:
                        st.info("No strong word-level signals found (text may be too short or highly novel).")
                else:
                    st.warning("Train the classifier first (`python scripts/train_category_classifier.py`).")

    # ────────────────────────────────────────────────────────────────────
    # D. Fine-tuned DistilBERT (Evaluation / Comparison Only)
    # ────────────────────────────────────────────────────────────────────
    if distilbert:
        st.divider()
        st.markdown("#### Fine-tuned DistilBERT  _(Evaluation / Dissertation Comparison Only)_")
        st.caption(
            "distilbert-base-uncased fine-tuned for 8 epochs (early stopping patience=2, best checkpoint "
            "at epoch 7, val acc 62.4%). Weighted cross-entropy loss. Max sequence length: 128 tokens. "
            "**Not in the live pipeline** — at ~700 training samples TF-IDF + LR matches its accuracy "
            "while loading approximately 20× faster."
        )

        db_col1, db_col2 = st.columns([1, 1])
        with db_col1:
            df_db = _per_class_df(distilbert["report"], cat_labels)
            st.dataframe(df_db, use_container_width=True, hide_index=True)
            macro_db = distilbert["report"]["macro avg"]
            st.markdown(f"""
| Metric | Value |
|---|---|
| **Accuracy** | {db_acc*100:.1f}% |
| **Macro Precision** | {macro_db['precision']*100:.1f}% |
| **Macro Recall** | {macro_db['recall']*100:.1f}% |
| **Macro F1** | {macro_db['f1-score']*100:.1f}% |
""")
            df_db_sorted = df_db.sort_values("F1-Score", ascending=True)
            df_db_sorted["F1 (%)"] = (df_db_sorted["F1-Score"] * 100).round(1)
            fig_db = px.bar(df_db_sorted, x="F1 (%)", y="Class", orientation="h",
                            title="DistilBERT — F1 per Category",
                            color_discrete_sequence=[C_BLUE],
                            text=df_db_sorted["F1 (%)"].apply(lambda v: f"{v:.1f}%"))
            fig_db.update_traces(textposition="outside")
            fig_db.update_layout(**_LAYOUT,
                                 xaxis=dict(range=[0, 115], ticksuffix="%", automargin=True),
                                 yaxis=dict(automargin=True))
            st.plotly_chart(fig_db, use_container_width=True)
        with db_col2:
            st.plotly_chart(
                _confusion_heatmap(distilbert["confusion_matrix"],
                                   short_labels, "DistilBERT Confusion Matrix"),
                use_container_width=True,
            )

        delta_vs_zs = (db_acc - zs_acc) * 100
        if trained and tr_acc is not None:
            delta_vs_lr = (db_acc - tr_acc) * 100
            if db_acc >= tr_acc:
                st.success(
                    f"DistilBERT achieves **{db_acc*100:.1f}% accuracy** and **{db_f1*100:.1f}% macro F1** "
                    f"— **{delta_vs_zs:+.1f}%** over zero-shot and **{delta_vs_lr:+.1f}%** over TF-IDF + LR. "
                    "Contextual embeddings resolve category ambiguities that surface-level n-gram features cannot."
                )
            else:
                st.info(
                    f"DistilBERT achieves **{db_acc*100:.1f}% accuracy** — **{delta_vs_zs:+.1f}%** over "
                    f"zero-shot but **{abs(delta_vs_lr):.1f}% below** TF-IDF + LR ({tr_acc*100:.1f}%). "
                    "At ~700 training samples, TF-IDF + LR generalises more effectively than the transformer. "
                    "Transformers typically need >2,000 samples to outperform strong linear baselines — "
                    "a well-documented threshold effect and a key dissertation observation."
                )
        else:
            st.info(
                f"DistilBERT achieves **{db_acc*100:.1f}% accuracy** — "
                f"**{delta_vs_zs:+.1f}%** over the zero-shot baseline."
            )

# ---- Sentiment Analysis ----------------------------------------------
with tab3:
    st.markdown("### Sentiment Analysis (FinBERT)")
    st.caption(
        "ProsusAI/FinBERT evaluated on 30 hand-labelled samples: 12 negative, 12 neutral, 6 positive. "
        "FinBERT is pre-trained on financial text and well-suited for financial complaint sentiment."
    )
    sent_labels = ["negative", "neutral", "positive"]
    col1, col2 = st.columns([1, 1])
    with col1:
        df_s = _per_class_df(R["sentiment_analysis"]["report"], sent_labels)
        st.dataframe(df_s, use_container_width=True, hide_index=True)
        macro = R["sentiment_analysis"]["report"]["macro avg"]
        st.markdown(f"""
| Metric | Value |
|---|---|
| **Accuracy** | {R['sentiment_analysis']['accuracy']*100:.1f}% |
| **Macro Precision** | {macro['precision']*100:.1f}% |
| **Macro Recall** | {macro['recall']*100:.1f}% |
| **Macro F1** | {macro['f1-score']*100:.1f}% |
""")
    with col2:
        fig = _confusion_heatmap(
            R["sentiment_analysis"]["confusion_matrix"],
            sent_labels, "Sentiment Confusion Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Observations**")
    st.success(
        "FinBERT achieves **100% accuracy** on the sentiment test set, correctly classifying all "
        "negative, neutral and positive samples. This reflects FinBERT's strong pre-training on "
        "financial domain text and the clear linguistic distinction between complaint-driven "
        "negativity and satisfaction-driven positivity in the test samples."
    )

# ---- RAG Retrieval ---------------------------------------------------
with tab4:
    st.markdown("### RAG Knowledge Base Retrieval")
    st.caption(
        "Sentence-transformer embeddings (all-MiniLM-L6-v2) + ChromaDB vector store evaluated on "
        "10 complaint queries. Precision@K measures whether the top-K retrieved articles contain "
        "a keyword relevant to the query's expected resolution category."
    )
    rag = R["rag_retrieval"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Precision@1", f"{rag['precision_at_1']*100:.1f}%")
    col2.metric("Precision@3", f"{rag['precision_at_3']*100:.1f}%")
    col3.metric("Test Queries", rag["n_test_cases"])

    rag_data = pd.DataFrame({
        "Metric": ["Precision@1", "Precision@3"],
        "Score (%)": [rag["precision_at_1"]*100, rag["precision_at_3"]*100],
    })
    fig = px.bar(rag_data, x="Metric", y="Score (%)", title="RAG Retrieval Precision",
                 color_discrete_sequence=[C_BLUE], text="Score (%)", range_y=[0, 110])
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(**_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Observations**")
    st.success(
        "RAG retrieval achieves **100% Precision@1 and @3** across all 10 test queries. "
        "The sentence-transformer embedding model effectively captures semantic similarity "
        "between complaint text and KB articles, retrieving contextually relevant documents "
        "even when the exact complaint vocabulary differs from article titles. "
        "The 30-article knowledge base covers all complaint categories with targeted resolution content."
    )

# ---- Risk Scoring ----------------------------------------------------
with tab5:
    st.markdown("### Complaint Risk Scoring")
    st.caption(
        "6-factor weighted risk scorer (rule-based) evaluated on 8 test cases spanning Critical, High, "
        "Medium and Low priority levels. ML risk scorer (Random Forest / Gradient Boosting trained on "
        "2,000 synthetic samples) shown alongside for comparison."
    )

    rs    = R["risk_scoring"]
    rs_ml = R.get("risk_scoring_ml")

    # KPIs
    if rs_ml:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rule-based Accuracy", f"{rs['accuracy']*100:.1f}%")
        k2.metric(f"ML Accuracy ({rs_ml.get('model_name','ML')})", f"{rs_ml['accuracy']*100:.1f}%",
                  delta=f"{(rs_ml['accuracy'] - rs['accuracy'])*100:+.1f}%")
        k3.metric("Test Cases", rs["n_test_cases"])
        k4.metric("ML Synthetic Test Acc",
                  f"{rs_ml.get('test_accuracy_training', 0)*100:.1f}%" if rs_ml.get('test_accuracy_training') else "—",
                  help="Accuracy on 400 random synthetic samples — measures how well ML learned rule-based boundaries")
    else:
        k1, k2 = st.columns(2)
        k1.metric("Priority Accuracy", f"{rs['accuracy']*100:.1f}%")
        k2.metric("Test Cases", rs["n_test_cases"])

    # Rule-based cases table
    st.markdown("#### Rule-based Risk Scorer")
    cases_df = pd.DataFrame(rs["cases"])
    cases_df["Correct"] = cases_df["expected"] == cases_df["predicted"]
    cases_df["score_pct"] = (cases_df["score"] * 100).round(1)
    cases_df = cases_df.rename(columns={
        "category": "Category", "tier": "Tier",
        "expected": "Expected Priority", "predicted": "Predicted Priority",
        "score_pct": "Risk Score (%)", "Correct": "Correct?"
    })
    st.dataframe(cases_df[["Category", "Tier", "Risk Score (%)",
                            "Expected Priority", "Predicted Priority", "Correct?"]],
                 use_container_width=True, hide_index=True)

    st.info(
        f"Rule-based scorer achieves **{rs['accuracy']*100:.1f}% priority accuracy** on boundary cases. "
        "Misclassifications occur at tier boundaries where the continuous risk score sits close to a "
        "threshold — e.g. a Billing Concern with moderate severity scores Medium instead of High. "
        "The scorer correctly separates Critical/High fraud and Platinum-tier complaints from "
        "lower-severity general inquiries."
    )

    # ML risk scorer comparison
    if rs_ml:
        st.divider()
        st.markdown(f"#### ML Risk Scorer — {rs_ml.get('model_name', 'ML')} _(Trained on Synthetic Data)_")
        st.caption(
            "Random Forest and Gradient Boosting classifiers trained on 2,000 synthetically generated "
            "risk factor combinations (rule-based labels as pseudo-labels). Features: category severity "
            "base, sentiment component, escalation/unresolved/frequency norms, customer value multiplier. "
            "Best model selected by 5-fold cross-validation on the training set."
        )

        # CV comparison chart
        cv = rs_ml.get("cv_results", {})
        if cv:
            cv_df = pd.DataFrame([
                {"Model": k, "CV Accuracy (%)": round(v["cv_mean"] * 100, 1),
                 "±": round(v["cv_std"] * 100, 1)}
                for k, v in cv.items()
            ])
            fig_cv = px.bar(cv_df, x="Model", y="CV Accuracy (%)",
                            title="5-fold CV Accuracy by Model",
                            color_discrete_sequence=[C_BLUE],
                            text=cv_df["CV Accuracy (%)"].apply(lambda v: f"{v:.1f}%"),
                            error_y="±")
            fig_cv.update_traces(textposition="outside")
            fig_cv.update_layout(**_LAYOUT, yaxis=dict(range=[0, 115], ticksuffix="%"))
            st.plotly_chart(fig_cv, use_container_width=True)

        # ML test cases
        ml_cases_df = pd.DataFrame(rs_ml["cases"])
        ml_cases_df["Correct"] = ml_cases_df["expected"] == ml_cases_df["predicted"]
        ml_cases_df = ml_cases_df.rename(columns={
            "category": "Category", "tier": "Tier",
            "expected": "Expected Priority", "predicted": "Predicted Priority",
            "Correct": "Correct?"
        })
        st.dataframe(ml_cases_df[["Category", "Tier", "Expected Priority",
                                   "Predicted Priority", "Correct?"]],
                     use_container_width=True, hide_index=True)

        delta = (rs_ml["accuracy"] - rs["accuracy"]) * 100
        if rs_ml["accuracy"] >= rs["accuracy"]:
            st.success(
                f"ML scorer achieves **{rs_ml['accuracy']*100:.1f}% accuracy** on the 8-case evaluation "
                f"set — **{delta:+.1f}%** over the rule-based baseline. "
                "The model learns priority boundaries directly from feature distributions rather than "
                "relying on manually set thresholds, making it more robust to boundary-case complaints."
            )
        else:
            st.warning(
                f"ML scorer achieves **{rs_ml['accuracy']*100:.1f}% accuracy** on the 8 hand-crafted boundary cases "
                f"(**{delta:+.1f}%** vs rule-based 62.5%). "
                "These 8 cases all have `escalation_count=0`; the training distribution rarely produces Critical "
                "with zero escalations, so the ML model under-predicts Critical. "
                f"On the broader synthetic test set (400 random samples) the model achieves "
                f"**{rs_ml.get('test_accuracy_training', 0)*100:.1f}% accuracy**, confirming it learns the "
                "rule-based boundaries well in general. The divergence here is a dissertation finding: "
                "ML trained on pseudo-labels inherits the rule-based system's feature correlations and "
                "can differ from human expert expectations on edge cases."
            )

st.divider()

# ---- Evaluation Methodology note ------------------------------------
st.markdown("### Evaluation Methodology")
st.caption(
    "All test sets are hand-labelled by the researcher. Test samples were constructed to be independent "
    "of the training/simulation data. Metrics are computed using scikit-learn's classification_report "
    f"with zero_division=0. Evaluation runtime: {R.get('elapsed_seconds', '—')}s."
)
with st.expander("Test Set Sizes"):
    st.markdown("""
| Component | Test Samples | Classes |
|---|---|---|
| Intent Detection | 50 | 7 |
| Category Classification | 45 | 9 |
| Sentiment Analysis | 30 | 3 |
| RAG Retrieval | 10 queries | — |
| Risk Scoring | 8 cases | 4 priority levels |
""")
