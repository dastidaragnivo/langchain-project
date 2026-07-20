"""
ReflexRAG — Self-Correcting Agentic RAG
=========================================
Streamlit front-end for the LangGraph agentic RAG pipeline defined in
`graph/graph.py`. The pipeline retrieves from a local vector store, grades
document relevance, generates an answer, checks the answer for hallucination
and query-relevance, and falls back to live web search (Tavily) when the
local knowledge base isn't good enough — looping until it produces a
grounded, useful answer.

Run with:
    streamlit run streamlit_app.py
"""

import io
import sys
import contextlib
from pathlib import Path

import streamlit as st

# ----------------------------------------------------------------------------
# App identity
# ----------------------------------------------------------------------------
APP_NAME = "ReflexRAG"
APP_TAGLINE = "Self-correcting agentic RAG — it checks its own work."

# ----------------------------------------------------------------------------
# Make the repo root importable (mirrors main.py) and import the graph
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_graph():
    """Import and compile the LangGraph workflow once per server process."""
    from dotenv import load_dotenv
    load_dotenv()
    from graph.graph import app as compiled_graph
    return compiled_graph


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(circle at 15% 0%, #1a1033 0%, #0e0b1a 45%, #0a0812 100%); }

    .rr-hero {
        padding: 1.6rem 1.8rem;
        border-radius: 16px;
        background: linear-gradient(120deg, rgba(124,58,237,0.28), rgba(56,189,248,0.14));
        border: 1px solid rgba(167,139,250,0.35);
        margin-bottom: 1.4rem;
    }
    .rr-hero h1 {
        margin: 0;
        font-size: 2.1rem;
        background: linear-gradient(90deg, #c4b5fd, #93c5fd, #67e8f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .rr-hero p { margin: 0.35rem 0 0 0; color: #cbd5e1; font-size: 0.98rem; }

    .rr-badge {
        display: inline-block;
        padding: 0.15rem 0.65rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-top: 0.6rem;
        border: 1px solid rgba(255,255,255,0.15);
        color: #e2e8f0;
        background: rgba(255,255,255,0.06);
    }

    .rr-step {
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
        padding: 0.5rem 0.7rem;
        border-radius: 10px;
        margin-bottom: 0.4rem;
        border-left: 3px solid transparent;
        font-size: 0.88rem;
        line-height: 1.35;
    }
    .rr-step-info    { background: rgba(148,163,184,0.08); border-left-color: #64748b; color: #cbd5e1; }
    .rr-step-action  { background: rgba(56,189,248,0.10); border-left-color: #38bdf8; color: #bae6fd; }
    .rr-step-good    { background: rgba(74,222,128,0.10); border-left-color: #4ade80; color: #bbf7d0; }
    .rr-step-warn    { background: rgba(250,204,21,0.10); border-left-color: #facc15; color: #fef08a; }
    .rr-step-loop    { background: rgba(244,114,182,0.10); border-left-color: #f472b6; color: #fbcfe8; }
    .rr-step-final   { background: rgba(167,139,250,0.14); border-left-color: #a78bfa; color: #ede9fe; }

    .rr-answer {
        padding: 1.1rem 1.3rem;
        border-radius: 14px;
        background: rgba(167,139,250,0.08);
        border: 1px solid rgba(167,139,250,0.3);
        color: #f1f5f9;
        line-height: 1.55;
    }

    .rr-source-chip {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 8px;
        background: rgba(56,189,248,0.12);
        border: 1px solid rgba(56,189,248,0.3);
        color: #bae6fd;
        font-size: 0.78rem;
        margin: 0.15rem 0.3rem 0.15rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Trace-line interpretation
# Maps substrings from the pipeline's stdout prints -> (css class, icon, label)
# ----------------------------------------------------------------------------
TRACE_RULES = [
    ("---CONTEXTUALIZE QUESTION---", "rr-step-info", "🧵", "Checking the question against prior conversation"),
    ("---NO PRIOR HISTORY, USING QUESTION AS-IS---", "rr-step-info", "🆕", "First question this session — using it as-is"),
    ("---REWRITTEN QUESTION:", "rr-step-good", "✏️", "Rewrote as a standalone question using chat history"),
    ("---RETRIEVE---", "rr-step-action", "📚", "Retrieving documents from the local knowledge base"),
    ("---CHECK DOCUMENT RELEVANCE TO QUESTION---", "rr-step-info", "🔍", "Grading each retrieved document for relevance"),
    ("---GRADE: DOCUMENT RELEVANT---", "rr-step-good", "✅", "Document marked relevant"),
    ("---GRADE: DOCUMENT NOT RELEVANT---", "rr-step-warn", "⚠️", "Document marked not relevant — discarded"),
    ("WERE RELEVANT (FEWER THAN HALF)---", "rr-step-warn", "🚫", "Fewer than half the retrieved documents were relevant"),
    ("---ASSESS GRADED DOCUMENTS---", "rr-step-info", "🧭", "Deciding whether local context is sufficient"),
    ("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT", "rr-step-loop", "🌐", "Local docs insufficient → falling back to live web search"),
    ("---DECISION: DOCUMENTS ARE RELEVANT", "rr-step-good", "✨", "Local docs sufficient → proceeding straight to generation"),
    ("---WEB SEARCH---", "rr-step-action", "🌐", "Searching the web (Tavily) for supplementary context"),
    ("---GENERATE---", "rr-step-action", "🤖", "Generating an answer grounded in the retrieved context"),
    ("---CHECK HALLUCINATIONS---", "rr-step-info", "🕵️", "Checking whether the answer is grounded in the documents"),
    ("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---", "rr-step-good", "✅", "Answer is grounded — no hallucination detected"),
    ("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---", "rr-step-loop", "♻️", "Answer not grounded → regenerating"),
    ("---GRADE GENERATION vs QUESTION---", "rr-step-info", "🎯", "Checking whether the answer actually resolves the question"),
    ("---DECISION: GENERATION ANSWERS QUESTION---", "rr-step-final", "🎉", "Answer resolves the question — done"),
    ("---DECISION: GENERATION DOES NOT ANSWER QUESTION---", "rr-step-loop", "🔁", "Answer doesn't resolve the question → retrying via web search"),
]


def render_trace(raw_stdout: str) -> list[str]:
    """Turn captured stdout into styled HTML trace lines. Returns raw lines too."""
    lines = [ln.strip() for ln in raw_stdout.splitlines() if ln.strip()]
    html_chunks = []
    for line in lines:
        matched = False
        for marker, css_class, icon, label in TRACE_RULES:
            if marker in line:
                html_chunks.append(
                    f'<div class="rr-step {css_class}"><span>{icon}</span><span>{label}</span></div>'
                )
                matched = True
                break
        if not matched:
            html_chunks.append(f'<div class="rr-step rr-step-info"><span>•</span><span>{line}</span></div>')
    return html_chunks


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### 🧠 {APP_NAME}")
    st.caption(APP_TAGLINE)

    st.markdown("---")
    st.markdown("#### How it works")
    st.markdown(
        """
1. **Retrieve** — pulls chunks from a local Chroma vector store
2. **Grade** — an LLM judges each chunk's relevance
3. **Generate** — drafts an answer from relevant context
4. **Hallucination check** — is the answer grounded in the docs?
5. **Answer check** — does it actually resolve the question?
6. **Web search fallback** — kicks in automatically if local context
   is weak or the answer fails a check, then loops back
        """
    )

    st.markdown("---")
    st.markdown("#### Knowledge base")
    st.caption("Indexed sources (see `ingestion.py`):")
    kb_sources = [
        ("LLM Powered Autonomous Agents", "lilianweng.github.io"),
        ("Prompt Engineering", "lilianweng.github.io"),
        ("Adversarial Attacks on LLMs", "lilianweng.github.io"),
    ]
    for title, domain in kb_sources:
        st.markdown(f'<span class="rr-source-chip">📄 {title}</span>', unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("⚙️ Required environment variables"):
        st.code(
            "ANTHROPIC_API_KEY=...\n"
            "VOYAGEAI_API_KEY=...\n"
            "TAVILY_API_KEY=...\n"
            "LANGSMITH_API_KEY=...   # used to pull the RAG prompt",
            language="bash",
        )

    st.markdown("---")
    show_trace = st.toggle("Show execution trace", value=True)
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.pop("chat_history", None)
        st.rerun()

# ----------------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="rr-hero">
        <h1>🧠 {APP_NAME}</h1>
        <p>{APP_TAGLINE}</p>
        <div>
            <span class="rr-badge">⚡ LangGraph</span>
            <span class="rr-badge">🎯 Corrective RAG</span>
            <span class="rr-badge">🔁 Self-grading loop</span>
            <span class="rr-badge">🌐 Web search fallback</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Load the graph
# ----------------------------------------------------------------------------
try:
    graph_app = load_graph()
    load_error = None
except Exception as exc:  # noqa: BLE001
    graph_app = None
    load_error = exc

if load_error is not None:
    st.error(
        "Couldn't initialize the agent pipeline. Check that your API keys are set "
        "and the local Chroma index exists (run `ingestion.py` first)."
    )
    with st.expander("Error details"):
        st.exception(load_error)
    st.stop()

# ----------------------------------------------------------------------------
# Chat state
# ----------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"question": str, "answer": str, "trace": list[str]}

EXAMPLE_QUESTIONS = [
    "What are the types of agent memory?",
    "What is prompt engineering?",
    "What are common adversarial attacks on LLMs?",
]

if not st.session_state.chat_history:
    st.markdown("##### Try an example, or ask your own question below")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, ex_q in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(ex_q, use_container_width=True):
            st.session_state["pending_question"] = ex_q

# Render past turns
for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        if show_trace and turn.get("trace"):
            with st.expander("🔬 Execution trace", expanded=False):
                st.markdown("\n".join(turn["trace"]), unsafe_allow_html=True)
        st.markdown(f'<div class="rr-answer">{turn["answer"]}</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Input handling
# ----------------------------------------------------------------------------
pending = st.session_state.pop("pending_question", None)
user_question = st.chat_input("Ask a question about the indexed sources...") or pending

if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        trace_placeholder = st.expander("🔬 Execution trace", expanded=True) if show_trace else None
        trace_html_container = trace_placeholder.empty() if trace_placeholder else None
        answer_placeholder = st.empty()

        # Prior turns as (question, answer) pairs, so the graph can rewrite
        # follow-up questions (e.g. "what about its downsides?") into
        # standalone ones before retrieval.
        history_pairs = [
            (turn["question"], turn["answer"]) for turn in st.session_state.chat_history
        ]

        with st.spinner("Thinking through retrieve → grade → generate → verify..."):
            buffer = io.StringIO()
            try:
                with contextlib.redirect_stdout(buffer):
                    result = graph_app.invoke(
                        input={"question": user_question, "chat_history": history_pairs}
                    )
                error = None
            except Exception as exc:  # noqa: BLE001
                result = None
                error = exc

        trace_lines = render_trace(buffer.getvalue())
        if trace_html_container is not None:
            trace_html_container.markdown("\n".join(trace_lines), unsafe_allow_html=True)

        if error is not None:
            answer_placeholder.error(f"Something went wrong while running the pipeline: {error}")
            answer_text = f"⚠️ Error: {error}"
        else:
            answer_text = (result or {}).get("generation", "_No answer was generated._")
            answer_placeholder.markdown(f'<div class="rr-answer">{answer_text}</div>', unsafe_allow_html=True)

    st.session_state.chat_history.append(
        {"question": user_question, "answer": answer_text, "trace": trace_lines}
    )