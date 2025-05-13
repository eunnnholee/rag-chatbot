from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 법률 이름 매핑 딕셔너리
LAW_NAME_MAPPING = {
    "고용보험법": "employment_insurance_act",
    "근로기준법": "labor_standards_act",
    "남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률": "gender_equality_employment_act",
}


def get_collection_name(law_name: str) -> str:
    """한글 법률명을 영문 컬렉션 이름으로 변환"""
    return LAW_NAME_MAPPING.get(law_name, law_name.replace(" ", "_").lower())


def create_vector_store(documents: list, collection_name: str) -> Chroma:
    """문서로부터 벡터 저장소 생성"""
    return Chroma.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(),
        collection_name=collection_name,
        persist_directory="./chroma_db",
    )


def load_vector_store(collection_name: str) -> Chroma:
    """기존 벡터 저장소 로드"""
    return Chroma(
        persist_directory="./chroma_db",
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(),
    )
