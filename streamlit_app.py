# streamlit_app.py
# Production RAG Chatbot — Streamlit Frontend
# Calls the FastAPI /query endpoint; sidebar controls retrieval mode + source filter.

import os
import json
import requests
import streamlit as st

# ─── CONFIG ───────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ─── PAGE SETUP ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Production RAG Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Main background ── */
    .stApp { background: #0d1117; color: #e6edf3; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #30363d;
    }

    /* ── Chat bubbles ── */
    .user-bubble {
        background: #1f6feb22;
        border: 1px solid #1f6feb55;
        border-radius: 12px 12px 4px 12px;
        padding: 12px 16px;
        margin: 8px 0 8px 60px;
        color: #e6edf3;
        font-size: 15px;
        line-height: 1.6;
    }
    .bot-bubble {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px 12px 12px 4px;
        padding: 12px 16px;
        margin: 8px 60px 8px 0;
        color: #c9d1d9;
        font-size: 15px;
        line-height: 1.6;
    }
    .bot-bubble strong { color: #58a6ff; }

    /* ── Source card ── */
    .source-card {
        background: #0d1117;
        border: 1px solid #21262d;
        border-left: 3px solid #1f6feb;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 13px;
        color: #8b949e;
    }
    .source-card .meta {
        color: #1f6feb;
        font-weight: 600;
        margin-bottom: 4px;
        font-size: 12px;
    }
    .source-card .chunk-text {
        color: #c9d1d9;
        font-size: 13px;
        line-height: 1.5;
    }

    /* ── Mode badge ── */
    .mode-badge {
        display: inline-block;
        background: #1f6feb22;
        border: 1px solid #1f6feb44;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 11px;
        color: #58a6ff;
        font-weight: 600;
        margin-bottom: 8px;
    }

    /* ── RAGAS score row ── */
    .ragas-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid #21262d;
        font-size: 13px;
    }
    .ragas-score {
        font-weight: 700;
        color: #3fb950;
    }
    .ragas-score.mid { color: #d29922; }
    .ragas-score.low { color: #f85149; }

    /* ── Hide Streamlit branding ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Input ── */
    .stTextInput > div > div > input {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── HELPERS ──────────────────────────────────────────────────────────────────
MODE_LABELS = {
    "naive":        "Naive (Dense FAISS)",
    "hyde":         "HyDE (Hypothetical Doc)",
    "multi":        "Multi-Query (RRF)",
    "multi_rerank": "Hybrid (RRF + Cross-Encoder Rerank)",
}

METRIC_LABELS = {
    "faithfulness":      "Faithfulness",
    "answer_relevancy":  "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall":    "Context Recall",
}


def check_health() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def fetch_sources() -> list[str]:
    try:
        r = requests.get(f"{API_BASE}/sources", timeout=5)
        if r.status_code == 200:
            return r.json().get("sources", [])
    except requests.exceptions.ConnectionError:
        pass
    return []


def fetch_results() -> dict:
    try:
        r = requests.get(f"{API_BASE}/results", timeout=5)
        if r.status_code == 200:
            return r.json().get("results", {})
    except requests.exceptions.ConnectionError:
        pass
    return {}


def query_api(question: str, retrieval_mode: str, top_k: int) -> dict | None:
    payload = {
        "question": question,
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
    }
    try:
        r = requests.post(f"{API_BASE}/query", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the FastAPI backend. Is it running?")
    except requests.exceptions.Timeout:
        st.error("Request timed out. The backend may be overloaded.")
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e}")
    return None


def score_color(val: float) -> str:
    if val >= 0.75:
        return "ragas-score"
    elif val >= 0.50:
        return "ragas-score mid"
    return "ragas-score low"


def render_ragas_panel(results: dict):
    if not results:
        st.info("No cached evaluation results found. Run `python run_eval.py` to generate scores.")
        return

    for mode, scores in results.items():
        st.markdown(f"**{MODE_LABELS.get(mode, mode)}**")
        if isinstance(scores, dict):
            for metric, value in scores.items():
                if isinstance(value, (int, float)):
                    label = METRIC_LABELS.get(metric, metric)
                    css = score_color(float(value))
                    st.markdown(
                        f'<div class="ragas-row">'
                        f'<span>{label}</span>'
                        f'<span class="{css}">{value:.4f}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        st.markdown("---")


def render_sources(meta: list[dict], source_filter: str):
    """Render retrieved chunk cards, optionally filtered by source PDF."""
    for chunk_meta in meta:
        src = chunk_meta.get("source_file", "unknown")
        page = chunk_meta.get("page_number", "?")
        text = chunk_meta.get("text", "")  # may not be returned by API — shown if present

        # client-side filter
        if source_filter != "All Sources" and src != source_filter:
            continue

        st.markdown(
            f"""<div class="source-card">
                <div class="meta">{src} &nbsp;·&nbsp; Page {page}</div>
                {"<div class='chunk-text'>" + text[:400] + ("…" if len(text) > 400 else "") + "</div>" if text else ""}
            </div>""",
            unsafe_allow_html=True,
        )


# ─── SESSION STATE ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {role, content, meta, mode}

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## RAG Configuration")

    # ── Backend health indicator ───────────────────────────────────────────────
    _healthy = check_health()
    if _healthy:
        st.markdown(
            '<div style="display:inline-flex;align-items:center;gap:6px;'
            'background:#1a3a1a;border:1px solid #3fb95077;border-radius:20px;'
            'padding:3px 12px;font-size:12px;color:#3fb950;margin-bottom:8px;">'
            'Backend connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="display:inline-flex;align-items:center;gap:6px;'
            'background:#3a1a1a;border:1px solid #f8514977;border-radius:20px;'
            'padding:3px 12px;font-size:12px;color:#f85149;margin-bottom:8px;">'
            'Backend offline</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        list(MODE_LABELS.keys()),
        format_func=lambda x: MODE_LABELS[x],
        help=(
            "**Naive**: Direct FAISS similarity search.\n\n"
            "**HyDE**: Generates a hypothetical answer, then retrieves.\n\n"
            "**Multi-Query**: Expands query, merges via RRF.\n\n"
            "**Hybrid**: Multi-Query + cross-encoder reranking."
        ),
    )

    top_k = st.slider(
        "Top-K Chunks",
        min_value=1, max_value=10, value=5,
        help="Number of chunks to retrieve from the FAISS index.",
    )

    st.markdown("---")

    # ── Source Filter ──────────────────────────────────────────────────────────
    st.markdown("### Source Filter")
    all_sources = fetch_sources()
    if all_sources:
        selected_source = st.selectbox(
            "Filter displayed chunks by PDF",
            ["All Sources"] + all_sources,
        )
    else:
        selected_source = "All Sources"
        st.caption("No indexed PDFs found. Ingest a document first.")

    st.markdown("---")

    # ── RAGAS Evaluation Scores ────────────────────────────────────────────────
    with st.expander("RAGAS Evaluation Scores", expanded=False):
        ragas_results = fetch_results()
        render_ragas_panel(ragas_results)

    st.markdown("---")

    # ── Clear History ─────────────────────────────────────────────────────────
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"Backend: `{API_BASE}`")


# ─── MAIN AREA ────────────────────────────────────────────────────────────────
st.markdown("# Production RAG Chatbot")
st.markdown(
    f'<div class="mode-badge">Mode: {MODE_LABELS[retrieval_mode]}</div>',
    unsafe_allow_html=True,
)

# ── Chat History ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="bot-bubble"><strong>Answer</strong><br>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        if msg.get("meta"):
            with st.expander(
                f"Retrieved Chunks ({len(msg['meta'])} results) — Mode: {MODE_LABELS.get(msg.get('mode',''), msg.get('mode',''))}",
                expanded=False,
            ):
                render_sources(msg["meta"], selected_source)

# ── Input Bar ─────────────────────────────────────────────────────────────────
st.markdown("---")

with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Ask a question about your documents…",
            label_visibility="collapsed",
            placeholder="e.g. What is the main contribution of this paper?",
        )
    with col2:
        submitted = st.form_submit_button("Send", use_container_width=True)

if submitted and user_input.strip():
    # Record user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Call the API
    with st.spinner("Retrieving and generating…"):
        result = query_api(user_input, retrieval_mode, top_k)

    if result:
        answer = result.get("answer", "No answer returned.")
        meta   = result.get("sources", [])

        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "meta":    meta,
            "mode":    retrieval_mode,
        })

    st.rerun()

# ── Empty state hint ──────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        """
        <div style="text-align:center; color:#484f58; margin-top:80px;">
            <div style="font-size:18px; margin-top:12px;">Ask anything about your indexed documents</div>
            <div style="font-size:13px; margin-top:6px;">Select a retrieval mode in the sidebar, then type your question below.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
