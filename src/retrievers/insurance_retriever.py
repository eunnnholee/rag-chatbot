import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.vector_db.vector_store import get_collection_name


def get_insurance_retriever():
    embeddings = OpenAIEmbeddings()
    collection_name = get_collection_name("고용보험법")

    # Chroma DB 디렉토리 확인
    if not os.path.exists("./chroma_db"):
        raise ValueError("Chroma DB directory not found at ./chroma_db")

    vector_db = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db",
        collection_name=collection_name,
    )

    # 컬렉션 존재 여부 확인
    if not vector_db._collection:
        raise ValueError(f"Collection '{collection_name}' not found in Chroma DB")

    return vector_db.as_retriever(search_kwargs={"k": 5})
