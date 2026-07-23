import asyncio
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_core.messages import HumanMessage
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
    async with stdio_client(stdio_server_params) as (read,write):
        print("Client connected to math server!")
        async with ClientSession(read_stream=read, write_stream=write) as session:
            await session.initialize()
            print("session initialized!")
            tools = await load_mcp_tools(session)

            # Creates a Langgraph ReAct agent with the MCP tools from the math server
            # Host application will interact with this agent
            agent = create_agent(llm, tools)

            result = await agent.ainvoke({"messages": [HumanMessage(content=input("Enter a math question: "))]})
            result_text = result["messages"][-1].content
            print(result_text.replace("**", "").strip())  # Remove any extra asterisks from the output

        
if __name__ == "__main__":
    asyncio.run(main())
