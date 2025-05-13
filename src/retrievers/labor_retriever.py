from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.vector_db.vector_store import get_collection_name


def get_labor_retriever():
    embeddings = OpenAIEmbeddings()
    vector_db = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db",
        collection_name=get_collection_name("근로기준법"),
    )
    return vector_db.as_retriever(search_kwargs={"k": 5})
