import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_voyageai import VoyageAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    loader = TextLoader("mediumblog1.txt", encoding="utf8")
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks of text.")

    embeddings = VoyageAIEmbeddings(model="voyage-large-2", api_key=os.getenv("VOYAGE_API_KEY"))

    print("Ingesting...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("INDEX_NAME"))

    print("Ingestion complete.")