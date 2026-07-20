from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question.
    Web search is triggered only if FEWER THAN HALF of the retrieved
    documents are relevant -- if at least half are relevant, we proceed to
    generate using the relevant subset. This avoids discarding a mostly-good
    retrieval (e.g. 3 of 4 chunks relevant) just because one chunk missed,
    while still catching genuinely weak retrievals (e.g. only 1 of 4
    relevant) that a "zero relevant" threshold would let through unchecked.

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
        print("---RETRIEVER RETURNED ZERO DOCUMENTS FOR THIS QUERY---")
        return {"documents": [], "question": question, "web_search": True}

    grader_inputs = [{"question": question, "document": d.page_content} for d in documents]
    # max_concurrency caps parallel API calls to stay comfortably under
    # typical per-minute rate limits; raise/lower to taste.
    scores = retrieval_grader.batch(grader_inputs, config={"max_concurrency": 5})

    filtered_docs = []
    for doc, score in zip(documents, scores):
        if score.binary_score.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            snippet = " ".join(doc.page_content.split())[:140]
            print(f"---REJECTED CHUNK PREVIEW: {snippet}...---")

    # Always log the ratio (not just when it triggers search) so the trace
    # makes it obvious whether this is a borderline call or a landslide.
    print(f"---RELEVANCE RATIO: {len(filtered_docs)}/{len(documents)} DOCUMENTS RELEVANT---")

    # Only fall back to web search when fewer than half the retrieved docs
    # were relevant -- a mostly-good retrieval (>= half relevant) is left
    # alone and generation proceeds on the relevant subset.
    web_search = len(filtered_docs) < (len(documents) / 2)
    if web_search:
        print("---DECISION: FEWER THAN HALF RELEVANT, FALLING BACK TO WEB SEARCH---")

    return {"documents": filtered_docs, "question": question, "web_search": web_search}