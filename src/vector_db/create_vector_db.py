import glob
import os
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from ..data_processing.law_parser import parse_law_with_sections
from .vector_store import create_vector_store, get_collection_name

# pdf 파일 목록을 확인
pdf_files = glob.glob(os.path.join("data", "*.pdf"))

# 법명 추출을 위한 정규표현식 패턴
law_name_pattern = r"^(.*?)\(법률\)"

for pdf_file in pdf_files:
    # 파일명에서 법명 추출
    filename = os.path.basename(pdf_file)
    match = re.search(law_name_pattern, filename)
    if match:
        law_name = match.group(1).strip()
        print(law_name)
        # 영문 컬렉션 이름으로 변환
        collection_name = get_collection_name(law_name)

    loader = PyPDFLoader(pdf_file)
    pages = loader.load()

    # 푸터 제거용 패턴
    footer_pattern = re.compile(r"법제처\s+\d+\s+국가법령정보센터\s*\n고용보험법")

    # 각 페이지에서 푸터 제거
    cleaned_pages = []
    for p in pages:
        text = p.page_content.strip()
        cleaned = footer_pattern.sub("", text)
        cleaned_pages.append(cleaned.strip())

    # 전체 문서 텍스트 결합
    law_text = "\n".join(cleaned_pages)

    parsed_law = parse_law_with_sections(law_text)

    final_docs = []

    for chapter, sections in parsed_law["장"].items():
        for section, articles in sections.items():
            for article_text in articles:
                # 조문 제목 줄 (예: "제1조(목적)")
                title_line = article_text.split("\n", 1)[0]

                # LangChain Document metadata 구성
                metadata = {
                    "source": pdf_file,
                    "chapter": chapter,
                    "section": section,
                    "name": law_name,
                    "article": title_line,
                }

                # LangChain Document 본문 구성
                content = (
                    f"[법률정보]\n"
                    f"다음 조항은 {metadata['name']} {metadata['chapter']} {metadata['section']}에서 발췌한 내용입니다.\n\n[법률조항]\n"
                    f"{article_text}"
                )

                # LangChain Document 객체 생성
                doc = Document(page_content=content, metadata=metadata)
                final_docs.append(doc)

    # 벡터 저장소 생성
    create_vector_store(final_docs, collection_name)
