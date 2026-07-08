import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from tavily import TavilyClient
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

#Using custom search tool in Tavily
@tool
def search(query: str) -> str:
    """
    A search tool that uses the TAVILY API to search the web.
    Args:
        query (str): The search query.
    Returns:
        str: The search results.
    """
    print(f"Searching for: {query}")
    return tavily.search(query=query)

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"))
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":[HumanMessage(content="search for 5 job postings in different companies for an ai engineer using langchain in the Kolkata area on linkedin and list their details")]})
    

    # configure_langsmith()


    print(result)


if __name__ == "__main__":
    main()
