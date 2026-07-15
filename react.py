#====================
#ReAct Agent's Brain
#====================

import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def triple(num:float)-> float:
    """
    param num: a number to triple
    returns: the triple of the input number
    """

    return float(num) * 3

tools = [TavilySearch(max_results = 1), triple]

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY")).bind_tools(tools)

