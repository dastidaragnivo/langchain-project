import asyncio
import atexit
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

from mcp_utils import ensure_weather_server

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"))

WEATHER_SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "servers", "weather_server.py")

# MultiServerMCPClient lets one client talk to several MCP servers at once,
# each with its own transport:
#   - math server: launched by the client itself as a subprocess (stdio)
#   - weather server: a remote HTTP server (sse). We auto-launch it below
#     (via mcp_utils.ensure_weather_server) so you don't have to start it
#     manually in a separate terminal every time.
client = MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            "args": ["C:\\Users\\dasti\\Python & Data Science\\langchain-course\\mcp-servers\\mcp-crash-course\\servers\\math_server.py"],
            "transport": "stdio",
        },
        "weather": {
            "url": "http://127.0.0.1:8000/sse",
            "transport": "sse",
        },
    }
)


async def main():
    # Auto-start weather_server.py if nothing is already listening on its port.
    # Returns None (and does nothing) if a server is already running there.
    weather_process = ensure_weather_server(WEATHER_SERVER_PATH)
    if weather_process:
        print("Auto-started weather_server.py in the background.")
        atexit.register(weather_process.terminate)
    else:
        print("Weather server already running — reusing it.")

    print("Connecting to MCP servers (math + weather)...")
    tools = await client.get_tools()
    print(f"Loaded {len(tools)} tool(s): {[t.name for t in tools]}")

    # Creates a Langgraph ReAct agent with tools pulled from BOTH MCP servers.
    # Host application will interact with this agent
    agent = create_agent(llm, tools)

    user_question = input("Ask a math or weather question: ")
    result = await agent.ainvoke({"messages": [HumanMessage(content=user_question)]})
    result_text = result["messages"][-1].content
    print(result_text.replace("**", "").strip())  # Remove any extra asterisks from the output


if __name__ == "__main__":
    asyncio.run(main())