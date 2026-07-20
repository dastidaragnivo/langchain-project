from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question.
    If any document is not relevant, we will set a flag to run web search.

    Documents are graded concurrently (via .batch) rather than one-by-one,
    since a synchronous for-loop was making one sequential Claude API round
    trip per retrieved document -- the single biggest source of latency in
    this node.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    if not documents:
        return {"documents": [], "question": question, "web_search": True}

    grader_inputs = [{"question": question, "document": d.page_content} for d in documents]
    # max_concurrency caps parallel API calls to stay comfortably under
    # typical per-minute rate limits; raise/lower to taste.
    scores = retrieval_grader.batch(grader_inputs, config={"max_concurrency": 5})

    filtered_docs = []
    web_search = False
    for doc, score in zip(documents, scores):
        if score.binary_score.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True

    return {"documents": filtered_docs, "question": question, "web_search": web_search}