import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"))

class GradeHallucinations(BaseModel):
    """Binary score for hallucination check on generated answer."""
    binary_score: bool = Field(description="Answer is grounded in the facts or not. 'yes' or 'no'.")

structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved facts: \n\n {documents} \n\n LLM generation: {generation}")
    ]
) 

hallucination_grader:RunnableSequence = hallucination_prompt | structured_llm_grader
