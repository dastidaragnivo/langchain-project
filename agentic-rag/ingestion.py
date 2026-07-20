import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_voyageai import VoyageAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Anchor to this file's own location rather than a relative "./.chroma"
# path. Relative paths resolve against the process's current working
# directory, which can differ between local runs and a deployed
# environment (e.g. Streamlit Cloud) -- if it doesn't match where the
# index actually lives, Chroma silently opens/creates an EMPTY collection
# at the wrong path instead of erroring, and every query returns zero
# documents.
BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIRECTORY = str(BASE_DIR / ".chroma")
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

_vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
)

# IMPORTANT: this only reads the already-persisted Chroma index from disk.
# It does NOT fetch or re-chunk the source URLs, so importing this module
# (e.g. `from ingestion import retriever` in graph/nodes/retrieve.py) is
# fast and has zero dependency on `unstructured`/spaCy at runtime. That's
# only needed when actually rebuilding the index via build_index() below.
retriever = _vectorstore.as_retriever()


def get_index_stats() -> dict:
    """
    Quick sanity check on the persisted index: how many chunks it holds and
    where it's reading from. Call this at app startup to catch an empty/
    misconfigured vectorstore immediately instead of discovering it one
    silent "web search fallback" at a time.
    """
    try:
        count = _vectorstore._collection.count()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "count": None, "path": PERSIST_DIRECTORY, "error": str(exc)}
    return {"ok": count > 0, "count": count, "path": PERSIST_DIRECTORY, "error": None}


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