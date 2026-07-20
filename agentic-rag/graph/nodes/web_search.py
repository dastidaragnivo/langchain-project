import os
from dotenv import load_dotenv
load_dotenv()

from typing import Any, Dict

from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

web_search_tool = TavilySearch(
    api_key=os.environ.get("TAVILY_API_KEY"),
    max_results=3
)

def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Performs a web search based on the question in the graph state.

    Args:
        state (GraphState): The current state of the graph.
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    
    tavily_response = web_search_tool.invoke({"query": question})
    tavily_results = tavily_response.get("results", []) if isinstance(tavily_response, dict) else []
    joined_tavily_result = "\n".join(
        tavily_result.get("content", "") for tavily_result in tavily_results
    )
    web_results = Document(page_content=joined_tavily_result)

    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    
    return {"documents": documents, "question": question}

if __name__ == "__main__":
    web_search(state={"question":"agent memory", "documents":None})