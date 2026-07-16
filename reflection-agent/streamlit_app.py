import html
import streamlit as st
from langchain_core.messages import HumanMessage

from main import graph  # reuses the compiled LangGraph from main.py

st.set_page_config(page_title="The Copy Desk — Tweet Reflection Agent", page_icon="✎", layout="centered")

# ----------------------------------------------------------------------------
# Design system: "The Copy Desk"
# A writer (generator) drafts, an editor (reflector) marks it up in red pen,
# the writer revises. Each pass is stacked like manuscript revision history.
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=JetBrains+Mono:wght@400;500;700&family=Caveat:wght@500;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --ink: #12161C;
        --panel: #1B222B;
        --panel-edge: #2A333F;
        --paper: #EDEFF2;
        --muted: #8A93A3;
        --amber: #E8A33D;
        --red: #C4433D;
    }

    .stApp {
        background: var(--ink);
        color: var(--paper);
    }

    /* Kill default Streamlit chrome that fights the design */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2.5rem; max-width: 760px; }

    /* ---------- Masthead ---------- */
    .masthead {
        border-bottom: 3px solid var(--paper);
        padding-bottom: 14px;
        margin-bottom: 6px;
    }
    .masthead .kicker {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.22em;
        color: var(--amber);
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .masthead h1 {
        font-family: 'Fraunces', serif;
        font-weight: 700;
        font-size: 2.6rem;
        line-height: 1.05;
        margin: 0;
        color: var(--paper);
    }
    .masthead .dek {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        color: var(--muted);
        margin-top: 8px;
    }
    .masthead .dek b { color: var(--paper); }

    /* ---------- Loop indicator ---------- */
    .loop-strip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--muted);
        letter-spacing: 0.05em;
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 18px 0 26px 0;
        flex-wrap: wrap;
    }
    .loop-strip .dot { color: var(--amber); }
    .loop-strip .dot.red { color: var(--red); }

    /* ---------- Revision cards ---------- */
    .card {
        border-radius: 3px;
        padding: 18px 20px;
        margin-bottom: 14px;
        position: relative;
        border-left: 4px solid var(--panel-edge);
        background: var(--panel);
    }
    .card.draft { border-left-color: var(--amber); }
    .card.critique { border-left-color: var(--red); background: #201A19; }

    .card .tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 10px;
        display: block;
    }
    .card.draft .tag { color: var(--amber); }
    .card.critique .tag { color: var(--red); }

    .card.draft .body {
        font-family: 'Inter', sans-serif;
        font-size: 1.02rem;
        line-height: 1.55;
        color: var(--paper);
        white-space: pre-wrap;
    }
    .card.critique .body {
        font-family: 'Caveat', cursive;
        font-size: 1.35rem;
        line-height: 1.4;
        color: #E8B8B2;
        white-space: pre-wrap;
    }

    /* ---------- Final approved stamp ---------- */
    .final-wrap {
        position: relative;
        margin-top: 8px;
        padding: 26px 24px 22px 24px;
        border: 2px solid var(--amber);
        border-radius: 4px;
        background: linear-gradient(180deg, rgba(232,163,61,0.08), rgba(232,163,61,0.02));
    }
    .final-wrap .tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--amber);
        margin-bottom: 12px;
        display: block;
    }
    .final-wrap .body {
        font-family: 'Fraunces', serif;
        font-size: 1.3rem;
        line-height: 1.5;
        color: var(--paper);
        white-space: pre-wrap;
    }
    .stamp {
        position: absolute;
        top: -14px;
        right: 18px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        color: var(--ink);
        background: var(--amber);
        padding: 5px 12px;
        border-radius: 2px;
        transform: rotate(3deg);
        font-weight: 700;
        box-shadow: 0 2px 6px rgba(0,0,0,0.35);
    }

    /* ---------- Input area ---------- */
    .stTextArea textarea {
        background: var(--panel) !important;
        color: var(--paper) !important;
        border: 1px solid var(--panel-edge) !important;
        font-family: 'Inter', sans-serif !important;
        border-radius: 3px !important;
    }
    .stTextArea label, .stSlider label {
        color: var(--muted) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
    }

    .stButton button {
        background: var(--amber) !important;
        color: var(--ink) !important;
        border: none !important;
        border-radius: 3px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        letter-spacing: 0.05em !important;
        padding: 0.6rem 1.4rem !important;
    }
    .stButton button:hover { background: #F0B25C !important; }
    .stButton button:disabled { background: var(--panel-edge) !important; color: var(--muted) !important; }

    section[data-testid="stSidebar"] { background: #0D1015; border-right: 1px solid var(--panel-edge); }
    section[data-testid="stSidebar"] * { color: var(--paper) !important; font-family: 'Inter', sans-serif !important; }

    .streamlit-expanderHeader {
        color: var(--muted) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Masthead
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="masthead">
        <div class="kicker">Vol. 1 — Generator vs. Reflector</div>
        <h1>The Copy Desk</h1>
        <div class="dek">A writer drafts. An editor marks it up in red pen. The writer revises —
        <b>until the copy is fit to print.</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("**How it works**")
    st.write(
        "Each round, the generator writes or revises a tweet, then the reflector "
        "critiques it as a margin note. The loop repeats until the message cap is hit."
    )
    st.markdown("---")
    st.markdown("**Session cap**")
    st.caption("Set in `main.py`'s `should_continue` — informational only here.")

if "history" not in st.session_state:
    st.session_state.history = None

topic = st.text_area(
    "Assignment",
    placeholder="e.g. Write a tweet announcing our new open-source LangGraph project",
    height=90,
)

run = st.button("SEND TO THE DESK →", disabled=not topic.strip())

if run:
    with st.spinner("Writer drafting, editor sharpening the red pen..."):
        try:
            result = graph.invoke({"messages": [HumanMessage(content=topic)]})
            st.session_state.history = result["messages"]
        except Exception as e:
            st.session_state.history = None
            st.error(f"The desk hit a snag: {e}")

messages = st.session_state.history

if messages:
    # Loop strip: quick visual tally of the back-and-forth
    strip_parts = []
    for i, msg in enumerate(messages):
        if i == 0 or msg.type == "ai":
            strip_parts.append('<span class="dot">●</span> DRAFT')
        else:
            strip_parts.append('<span class="dot red">●</span> EDIT')
    st.markdown(f'<div class="loop-strip">{"  →  ".join(strip_parts)}</div>', unsafe_allow_html=True)

    draft_no = 0
    critique_no = 0

    for i, msg in enumerate(messages):
        content = html.escape(msg.content)
        is_last = i == len(messages) - 1

        # First message = the original assignment/topic; subsequent ai = drafts; human = critiques
        if i == 0:
            draft_no += 1
            st.markdown(
                f"""
                <div class="card draft">
                    <span class="tag">Assignment — Brief №{draft_no}</span>
                    <div class="body">{content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif msg.type == "ai":
            draft_no += 1
            if is_last:
                st.markdown(
                    f"""
                    <div class="final-wrap">
                        <span class="stamp">✓ READY TO POST</span>
                        <span class="tag">Final Draft — №{draft_no}</span>
                        <div class="body">{content}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="card draft">
                        <span class="tag">Draft №{draft_no}</span>
                        <div class="body">{content}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            critique_no += 1
            st.markdown(
                f"""
                <div class="card critique">
                    <span class="tag">Editor's Note №{critique_no}</span>
                    <div class="body">{content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    st.markdown(
        '<p style="color:#8A93A3; font-family:Inter,sans-serif; margin-top:20px;">'
        "The desk is quiet. Hand in an assignment above to start the first round."
        "</p>",
        unsafe_allow_html=True,
    )