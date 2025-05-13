import os
from typing import List, Dict
from glob import glob
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from law_parser import LawDocumentParser
from policy_parser import PolicyDocumentParser  # 정책 파서 추후에 추가

load_dotenv()


class LawVectorDB:
    """
    FAISS 벡터 DB를 사용해 법령 및 정책 문서 검색을 지원하는 클래스.
    """

    def __init__(self, persist_dir: str = "./faiss_store"):
        """
        Args:
            persist_dir (str): 벡터 DB 저장 경로
        """
        self.persist_dir = persist_dir

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY가 .env에 설정되어 있지 않습니다.")
        self.embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

    def save_vector_db(self, docs: List[Document], db_name: str) -> None:
        """
        Document 리스트를 FAISS DB로 임베딩하고 저장합니다.
        """
        db = FAISS.from_documents(docs, self.embedding)
        db_path = os.path.join(self.persist_dir, db_name)
        os.makedirs(db_path, exist_ok=True)
        db.save_local(db_path)
        print(f"[SAVE] FAISS DB saved to {db_path}")

    def load_vector_db(self, db_name: str) -> FAISS:
        """
        저장된 FAISS DB를 불러옵니다.
        """
        db_path = os.path.join(self.persist_dir, db_name)
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"{db_path} 경로에 FAISS DB가 없습니다.")
        return FAISS.load_local(db_path, self.embedding)

    def similarity_search(self, query: str, db_name: str, k: int = 5) -> List[Document]:
        """
        주어진 쿼리로 벡터 검색을 수행합니다.
        """
        db = self.load_vector_db(db_name)
        results = db.similarity_search(query, k=k)
        return results

    def bulk_ingest_from_mapping(self, law_mapping: Dict[str, str], data_dir: str) -> None:
        """
        여러 개의 법령 PDF 파일을 law_mapping 기준으로 FAISS에 저장합니다.
        """
        pdf_files = glob(os.path.join(data_dir, '*.pdf'))
        pdf_files.sort()

        if len(pdf_files) != len(law_mapping):
            raise ValueError("PDF 파일 수와 매핑된 법령 수가 일치하지 않습니다.")

        for idx, (law_name, collection_name) in enumerate(law_mapping.items()):
            print(f"[LAW] {law_name} → {collection_name}")
            parser = LawDocumentParser(law_name)
            documents = parser.create_documents(pdf_files[idx])
            self.save_vector_db(documents, db_name=collection_name)

    def bulk_ingest_policies(self, policy_mapping: Dict[str, str], data_dir: str) -> None:
        """
        여러 개의 정책 PDF 파일을 policy_mapping 기준으로 FAISS에 저장합니다.
        """
        pdf_files = glob(os.path.join(data_dir, '*.pdf'))
        pdf_files.sort()

        if len(pdf_files) != len(policy_mapping):
            raise ValueError("정책 PDF 수와 매핑된 정책 수가 일치하지 않습니다.")

        for idx, (policy_name, collection_name) in enumerate(policy_mapping.items()):
            print(f"[POLICY] {policy_name} → {collection_name}")
            parser = PolicyDocumentParser(policy_name)
            documents = parser.create_documents(pdf_files[idx])
            self.save_vector_db(documents, db_name=collection_name)
