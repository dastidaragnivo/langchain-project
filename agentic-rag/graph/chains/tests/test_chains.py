import sys
import os
from pathlib import Path

from dotenv import load_dotenv
import pytest

# Ensure project root is importable before local package imports.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import GradeHallucinations, hallucination_grader
from ingestion import retriever

from pprint import pprint

def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory" #Example question
    docs = retriever.invoke(question)
    if not docs:
        pytest.skip("No documents in Chroma collection; ingest docs before running this integration test")
    doc_txt = docs[1].page_content if len(docs) > 1 else docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"document": doc_txt, "question": question}
    )

    assert res.binary_score.strip().lower() == "yes"


def test_retrieval_grader_answer_no() -> None:
    question = "agent memory" #Example question
    docs = retriever.invoke(question)
    if not docs:
        pytest.skip("No documents in Chroma collection; ingest docs before running this integration test")
    doc_txt = docs[1].page_content if len(docs) > 1 else docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"document": doc_txt, "question": "how to make a pizza"}
    )

    assert res.binary_score.strip().lower() == "no"

def test_generation_chain() -> None:
    question = "agent memory" #Example question
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context":docs, "question": question})
    pprint(generation)

def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory" #Example question
    docs = retriever.invoke(question)
    
    generation = generation_chain.invoke({"context":docs, "question": question})

    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )

    assert res.binary_score

def test_hallucination_grader_answer_no() -> None:
    question = "agent memory" #Example question
    docs = retriever.invoke(question)
    
    generation = generation_chain.invoke({"context":docs, "question": question})

    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": "In order to make pizza we need to start with the dough"}
    )

    assert not res.binary_score