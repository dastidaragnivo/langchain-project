# Langchain client which can connect to multiple servers at once
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import os

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"))

async def main():
    print("Hello from langchain_client.py!")

if __name__ == "__main__":
    asyncio.run(main())
    