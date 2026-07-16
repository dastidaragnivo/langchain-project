import html
import streamlit as st
from langchain_core.messages import HumanMessage

from main import graph  # reuses the compiled LangGraph from main.py

st.set_page_config(page_title="Tweet Reflection Agent", page_icon="🪶", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg: #14161A;
        --surface: #1D2026;
        --surface-2: #23262E;
        --accent: #7C5CFC;
        --accent-soft: rgba(124, 92, 252, 0.14);
        --text: #F1F2F6;
        --muted: #8B90A0;
    }

    .stApp { background: var(--bg); color: var(--text); }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 3rem; max-width: 680px; }

    .hero { text-align: center; margin-bottom: 8px; }
    .hero .badge {
        display: inline-block;
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        color: var(--accent);
        background: var(--accent-soft);
        padding: 5px 14px;
        border-radius: 999px;
        margin-bottom: 14px;
    }
    .hero h1 {
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        font-size: 2.1rem;
        margin: 0 0 8px 0;
        color: var(--text);
    }
    .hero p {
        font-family: 'Inter', sans-serif;
        color: var(--muted);
        font-size: 0.95rem;
        margin: 0 auto;
        max-width: 440px;
    }

    .bubble {
        border-radius: 18px;
        padding: 16px 20px;
        margin: 10px 0;
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        line-height: 1.55;
        color: var(--text);
        white-space: pre-wrap;
        box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    }
    .bubble.draft {
        background: var(--surface);
        border: 1px solid rgba(124, 92, 252, 0.25);
    }
    .bubble.critique {
        background: var(--surface-2);
        border: 1px solid rgba(255,255,255,0.06);
        margin-left: 24px;
    }
    .bubble .role {
        display: block;
        font-family: 'Manrope', sans-serif;
        font-weight: 700;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .bubble.draft .role { color: var(--accent); }
    .bubble.critique .role { color: var(--muted); }

    .final {
        border-radius: 18px;
        padding: 22px 24px;
        margin-top: 6px;
        background: linear-gradient(160deg, var(--accent-soft), transparent);
        border: 1.5px solid var(--accent);
        box-shadow: 0 8px 28px rgba(124, 92, 252, 0.18);
    }
    .final .role {
        display: flex;
        align-items: center;
        gap: 6px;
        font-family: 'Manrope', sans-serif;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 10px;
    }
    .final .body {
        font-family: 'Manrope', sans-serif;
        font-weight: 600;
        font-size: 1.2rem;
        line-height: 1.5;
        color: var(--text);
        white-space: pre-wrap;
    }

    .stTextArea textarea {
        background: var(--surface) !important;
        color: var(--text) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextArea label {
        color: var(--muted) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
    }
    .stButton button {
        background: var(--accent) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'Manrope', sans-serif !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 4px 14px rgba(124, 92, 252, 0.35) !important;
    }
    .stButton button:hover { filter: brightness(1.08); }
    .stButton button:disabled {
        background: var(--surface-2) !important;
        color: var(--muted) !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] { background: #101216; border-right: 1px solid rgba(255,255,255,0.05); }
    section[data-testid="stSidebar"] * { color: var(--text) !important; font-family: 'Inter', sans-serif !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <span class="badge">GENERATE → REFLECT → REFINE</span>
        <h1>Tweet Reflection Agent</h1>
        <p>Give it a topic. A writer drafts, an editor critiques, and the tweet
        improves round by round.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("**How it works**")
    st.write(
        "Each round: the writer drafts or revises a tweet, then the editor "
        "gives feedback. This repeats a few times before landing on a final version."
    )

if "history" not in st.session_state:
    st.session_state.history = None

topic = st.text_area(
    "What's the tweet about?",
    placeholder="e.g. Announce our new open-source LangGraph project",
    height=90,
)

run = st.button("Generate tweet", disabled=not topic.strip())

if run:
    with st.spinner("Writing, critiquing, revising..."):
        try:
            result = graph.invoke({"messages": [HumanMessage(content=topic)]})
            st.session_state.history = result["messages"]
        except Exception as e:
            st.session_state.history = None
            st.error(f"Something went wrong: {e}")

messages = st.session_state.history

if messages:
    for i, msg in enumerate(messages):
        content = html.escape(msg.content)
        is_last = i == len(messages) - 1

        if i == 0:
            st.markdown(
                f"""<div class="bubble draft"><span class="role">🎯 Topic</span>{content}</div>""",
                unsafe_allow_html=True,
            )
        elif msg.type == "ai":
            if is_last:
                st.markdown(
                    f"""
                    <div class="final">
                        <div class="role">✨ Final tweet</div>
                        <div class="body">{content}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<div class="bubble draft"><span class="role">✍️ Draft</span>{content}</div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f"""<div class="bubble critique"><span class="role">🖊️ Editor's note</span>{content}</div>""",
                unsafe_allow_html=True,
            )
else:
    st.markdown(
        '<p style="color:#8B90A0; font-family:Inter,sans-serif; text-align:center; margin-top:24px;">'
        "Enter a topic above to start."
        "</p>",
        unsafe_allow_html=True,
    )