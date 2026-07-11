import os
import sys
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_voyageai import VoyageAIEmbeddings
from langchain_pinecone import PineconeVectorStore
load_dotenv()

# Ensure stdout can print unicode safely in Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("Initializing components...")

embeddings = VoyageAIEmbeddings(model="voyage-large-2", api_key=os.getenv("VOYAGE_API_KEY"))
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"))

vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"),
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) #Uses 3 top chunks for context retrieval

prompt_template = ChatPromptTemplate.from_template(
    "Answer the question based only on the following context:\n\n{context}\n\nQuestion: {question}\n\nProvide a detailed answer:"
)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join([doc.page_content for doc in docs])

def retrieval_chain_without_lcel(query: str):
    """Perform retrieval and generate an answer without using LLM chain."""
    # Step-1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # Step-2: Format the retrieved documents into a single context string
    context = format_docs(docs)

    # Step-3: Create the prompt with context and query
    messages = prompt_template.format_messages(context=context, question=query)
    
    # Step-4: Generate the answer using the LLM
    response = llm.invoke(messages)
    
    return response.content

def create_retrieval_chain_with_lcel():
    """Create a retrieval chain that uses LLM and LCEL for structured processing."""

    # Create a RunnableSequence (sequence of Runnable lambda functions) that first retrieves documents and then generates an answer
    retrieval_chain = (
        RunnablePassthrough.assign(
            context = itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()  # Ensure the output is a string
    )

    return retrieval_chain

if __name__ == "__main__":
    print("Retrieving...")

    #Query
    query = "what is Pinecone in machine learning?"

    #===================================================================================================
    # Option 0: Raw invocation without RAG
    #===================================================================================================
    print('\n' * 20)
    print("IMPLEMENTATION 0: Raw invocation without RAG")
    print("="*70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    #===================================================================================================
    # Option 1: Retrieval-Augmented Generation (RAG) without LCEL (Langchain Expression Language)
    #===================================================================================================
    print('\n' * 20)
    print("IMPLEMENTATION 1: Retrieval-Augmented Generation (RAG) without LCEL")
    print("="*70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    #===================================================================================================
    # Option 2: Retrieval-Augmented Generation (RAG) with LCEL (Langchain Expression Language) - Better Approach
    #===================================================================================================
    print('\n' * 20)
    print("IMPLEMENTATION 2: Retrieval-Augmented Generation (RAG) with LCEL")
    print("="*70)
    retrieval_chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = retrieval_chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)