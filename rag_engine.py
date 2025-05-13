import os
from typing import List
from langchain_core.documents import Document
from langchain.chat_models import ChatOpenAI
# from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from vectorDB import LawVectorDB

load_dotenv()

class RAGEngine:
    """
    Prompt 기반으로 DB를 선택하고 RAG 응답을 생성하는 엔진 클래스.
    Agent를 사용하지 않고 Prompt를 활용함.
    """

    def __init__(self, persist_dir: str = "./faiss_store"):
        """
        RAGEngine 객체를 초기화합니다.

        Args:
            persist_dir (str): FAISS 벡터 DB가 저장된 디렉토리 경로
        """
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.vector_db = LawVectorDB(persist_dir=persist_dir)

        self.db_mapping = {
            "법령": ["employment_insurance", "labor_standards", "gender_equality_employment"],
            "정책": ["parenting_policy", "youth_policy"]
        }

    def classify_db(self, query: str) -> List[str]:
        """
        사용자의 질문을 분류하여 적절한 벡터 DB 목록을 반환합니다.

        Args:
            query (str): 사용자 입력 질문

        Returns:
            List[str]: 검색에 사용할 DB 이름 리스트
        """
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant for routing queries.

        Given the user's question, choose the appropriate database(s) to search from:
        - 법령 (legal database)
        - 정책 (policy database)
        - 둘다 (both)

        Respond with only one of the following:
        법령 / 정책 / 둘다

        User question:
        {query}

        Answer:
        """)
        response = self.llm.invoke(prompt.format_messages(query=query))
        result = response.content.strip()

        if "둘" in result:
            return self.db_mapping["법령"] + self.db_mapping["정책"]
        elif "정책" in result:
            return self.db_mapping["정책"]
        else:
            return self.db_mapping["법령"]

    def retrieve_documents(self, query: str, db_names: List[str], top_k: int = 5) -> List[Document]:
        """
        주어진 DB 목록에서 유사도 검색을 수행하여 관련 문서를 가져옵니다.

        Args:
            query (str): 사용자 질문
            db_names (List[str]): 검색 대상 DB 이름 리스트
            top_k (int): 각 DB에서 가져올 문서 수

        Returns:
            List[Document]: 검색된 문서 리스트
        """
        docs = []
        for db in db_names:
            docs += self.vector_db.similarity_search(query, db, k=top_k)
        return docs

    def generate_answer(self, query: str, docs: List[Document]) -> str:
        """
        검색된 문서를 기반으로 LLM을 사용해 최종 답변을 생성합니다.

        Args:
            query (str): 사용자 질문
            docs (List[Document]): 검색된 문서 리스트

        Returns:
            str: 생성된 자연어 응답
        """
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = ChatPromptTemplate.from_template("""
        You are an expert assistant that provides helpful answers based on provided legal or policy documents.

        ---
        Context:
        {context}
        ---

        Based on the above documents, answer the following question in korean:
        {query}
        """)
        response = self.llm.invoke(prompt.format_messages(query=query, context=context))
        return response.content.strip()

    def run(self, query: str) -> str:
        """
        전체 RAG 파이프라인을 실행합니다: 분류 → 문서 검색 → 응답 생성

        Args:
            query (str): 사용자 입력 질문

        Returns:
            str: 최종 응답 결과
        """
        db_names = self.classify_db(query)
        documents = self.retrieve_documents(query, db_names)
        answer = self.generate_answer(query, documents)
        return answer
