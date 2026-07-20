import os
from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"))
prompt = Client().pull_prompt(
	"rlm/rag-prompt",
	dangerously_pull_public_prompt=True,
)

generation_chain = prompt | llm | StrOutputParser()