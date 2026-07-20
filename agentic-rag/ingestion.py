import os
from dotenv import load_dotenv
from langchain_voyageai import VoyageAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PERSIST_DIRECTORY = "./.chroma"
COLLECTION_NAME = "rag-chroma"

URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

embeddings = VoyageAIEmbeddings(
    api_key=os.environ.get("VOYAGEAI_API_KEY"),
    model="voyage-3-large",
)

# IMPORTANT: this only reads the already-persisted Chroma index from disk.
# It does NOT fetch or re-chunk the source URLs, so importing this module
# (e.g. `from ingestion import retriever` in graph/nodes/retrieve.py) is
# fast and has zero dependency on `unstructured`/spaCy at runtime. That's
# only needed when actually rebuilding the index via build_index() below.
retriever = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
).as_retriever()


def build_index():
    """
    Fetch, chunk, and (re)index the source documents into Chroma.

    Run this manually and locally whenever you need to build or refresh the
    vector store:
        python ingestion.py

    This is intentionally NOT executed on import, so deploying the app
    (which only needs `retriever`) never triggers document loading.
    """
    from langchain_unstructured import UnstructuredLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = [
        UnstructuredLoader(
            web_url=url, chunking_strategy="basic", max_characters=1000000
        ).load()
        for url in URLS
    ]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    docs_splits = text_splitter.split_documents(docs_list)

    vectorstore = Chroma.from_documents(
        documents=docs_splits,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )
    print(f"Indexed {len(docs_splits)} chunks into '{PERSIST_DIRECTORY}'.")
    return vectorstore


if __name__ == "__main__":
    build_index()