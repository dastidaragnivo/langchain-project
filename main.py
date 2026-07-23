import asyncio
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"))

# MultiServerMCPClient lets one client talk to several MCP servers at once,
# each with its own transport:
#   - math server: launched by the client itself as a subprocess (stdio)
#   - weather server: an already-running remote HTTP server (sse)
#
# IMPORTANT: start weather_server.py in its own terminal BEFORE running this
# script (`python weather_server.py`). Unlike stdio, SSE does not get
# auto-launched by the client.
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
