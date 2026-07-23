"""
Nimbus & Numbers
-----------------
A Streamlit front end for the LangGraph ReAct agent that talks to two MCP
servers:
  - math_server.py    (stdio)  -> add, multiply
  - weather_server.py (sse)    -> get_weather (live Open-Meteo data)

Run:
    1. python weather_server.py          # in its own terminal, keep running
    2. streamlit run streamlit_app.py
"""

import asyncio
import atexit
import os
import sys

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

from mcp_utils import ensure_weather_server

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MATH_SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "servers", "math_server.py")
WEATHER_SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "servers", "weather_server.py")
WEATHER_SERVER_URL = "http://127.0.0.1:8000/sse"

MODES = {
    "🧮 Math": {
        "server_key": "math",
        "config": {
            "math": {
                "command": sys.executable,
                "args": [MATH_SERVER_PATH],
                "transport": "stdio",
            }
        },
        "placeholder": "e.g. What is 42 multiplied by 17, then add 8?",
        "hint": "Powered by the math_server MCP tool (add, multiply) over stdio.",
    },
    "⛅ Weather": {
        "server_key": "weather",
        "config": {
            "weather": {
                "url": WEATHER_SERVER_URL,
                "transport": "sse",
            }
        },
        "placeholder": "e.g. What's the weather like in Tokyo right now?",
        "hint": "Powered by the weather_server MCP tool (live Open-Meteo data) over SSE. Started automatically for you.",
    },
}

# --------------------------------------------------------------------------
# Page setup + styling
# --------------------------------------------------------------------------
st.set_page_config(page_title="Nimbus & Numbers", page_icon="🌦️", layout="centered")

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f4f7fb 0%, #eef2fa 100%);
        }
        .nn-hero {
            background: linear-gradient(135deg, #4f6ef7 0%, #7aa8ff 50%, #a5d8ff 100%);
            padding: 28px 32px;
            border-radius: 18px;
            color: white;
            margin-bottom: 22px;
            box-shadow: 0 8px 24px rgba(79, 110, 247, 0.25);
        }
        .nn-hero h1 {
            margin: 0;
            font-size: 2rem;
        }
        .nn-hero p {
            margin: 6px 0 0 0;
            opacity: 0.92;
        }
        .nn-answer-card {
            background: white;
            border-radius: 14px;
            padding: 20px 24px;
            border-left: 6px solid #4f6ef7;
            box-shadow: 0 4px 14px rgba(0,0,0,0.06);
            margin-top: 14px;
        }
        .nn-hint {
            font-size: 0.85rem;
            color: #6b7280;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="nn-hero">
        <h1>🌦️ Nimbus &amp; Numbers</h1>
        <p>One agent, two talents — crunch the math or check the skies.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Auto-launch the weather server (once per app session, survives reruns)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Starting weather server...")
def _boot_weather_server():
    process = ensure_weather_server(WEATHER_SERVER_PATH)
    if process:
        atexit.register(process.terminate)
        return "started"
    return "already_running"


try:
    _weather_status = _boot_weather_server()
except (RuntimeError, TimeoutError) as e:
    _weather_status = None
    st.warning(f"Couldn't auto-start the weather server: {e}\nWeather mode may not work until it's running.")

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    mode_label = st.radio("Choose what to ask about:", list(MODES.keys()))
    st.markdown("---")
    st.caption(MODES[mode_label]["hint"])
    st.markdown("---")
    if st.button("🧹 Clear history"):
        st.session_state.history = []
        st.rerun()

mode = MODES[mode_label]

if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------------------------------
# Agent runner
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )


async def ask_agent(question: str, server_config: dict):
    client = MultiServerMCPClient(server_config)
    tools = await client.get_tools()
    agent = create_agent(get_llm(), tools)
    result = await agent.ainvoke({"messages": [HumanMessage(content=question)]})
    return result["messages"]


def run_agent_sync(question: str, server_config: dict):
    return asyncio.run(ask_agent(question, server_config))


def extract_tool_calls(messages):
    """Pull out (tool_name, args, output) triples for display."""
    calls = []
    pending = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                pending[tc["id"]] = (tc["name"], tc["args"])
        elif isinstance(msg, ToolMessage):
            name, args = pending.get(msg.tool_call_id, (msg.name, {}))
            calls.append((name, args, msg.content))
    return calls


# --------------------------------------------------------------------------
# Main interaction
# --------------------------------------------------------------------------
with st.form("ask_form"):
    question = st.text_input(f"Ask your {mode_label.split()[-1].lower()} question:", placeholder=mode["placeholder"])
    submitted = st.form_submit_button("Ask ✨")

if submitted and question.strip():
    with st.spinner("Thinking..."):
        try:
            messages = run_agent_sync(question, mode["config"])
            answer = messages[-1].content.replace("**", "").strip()
            tool_calls = extract_tool_calls(messages)
            st.session_state.history.insert(0, {
                "mode": mode_label,
                "question": question,
                "answer": answer,
                "tool_calls": tool_calls,
            })
        except Exception as e:
            st.error(
                f"Something went wrong: {e}\n\n"
                "If you're in Weather mode, the auto-started server may not have come up in time — "
                "try refreshing the page."
            )
elif submitted:
    st.warning("Type a question first!")

# --------------------------------------------------------------------------
# History display
# --------------------------------------------------------------------------
for entry in st.session_state.history:
    st.markdown(
        f"""
        <div class="nn-answer-card">
            <div class="nn-hint">{entry['mode']} · "{entry['question']}"</div>
            <p style="margin-top:8px; font-size:1.05rem;">{entry['answer']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if entry["tool_calls"]:
        with st.expander("🔧 See tool calls"):
            for name, args, output in entry["tool_calls"]:
                st.markdown(f"**`{name}`** called with `{args}`")
                st.code(str(output), language="text")