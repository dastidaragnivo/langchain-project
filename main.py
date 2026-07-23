import asyncio
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"))

stdio_server_params = StdioServerParameters(
    command="python",
    args=["C:\\Users\\dasti\\Python & Data Science\\langchain-course\\mcp-servers\\mcp-crash-course\\servers\\math_server.py"],
    server_description="A simple math server that can add and multiply numbers.",
)

async def main():
    print("Hello from mcp-crash-course!")


if __name__ == "__main__":
    asyncio.run(main())
