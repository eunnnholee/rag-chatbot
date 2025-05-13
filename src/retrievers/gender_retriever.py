from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.vector_db.vector_store import get_collection_name


def get_gender_retriever():
    embeddings = OpenAIEmbeddings()
    vector_db = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db",
        collection_name=get_collection_name(
            "남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률"
        ),
    )
    return vector_db.as_retriever(search_kwargs={"k": 5})
