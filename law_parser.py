import re
from typing import List
from langchain_core.documents import Document
from langchain.document_loaders import PyPDFLoader


class LawDocumentParser:
    """
    법령 PDF 파일을 구조화된 LangChain Document 객체로 변환하는 클래스입니다.
    조문을 장/절/조문 단위로 분리하고, 메타데이터를 포함한 벡터 검색용 문서를 생성합니다.
    """

    def __init__(self, law_name: str):
        """
        클래스 초기화 메서드

        Args:
            law_name (str): PDF 파일 내에서 푸터 제거를 위한 법령 이름
        """
        self.law_name = law_name
        self.footer_pattern = re.compile(
            rf"법제처\s+\d+\s+국가법령정보센터\s*\n{re.escape(law_name)}"
        )

    def load_pdf(self, pdf_path: str) -> str:
        """
        PDF 파일을 불러와 푸터를 제거하고 하나의 텍스트 문자열로 반환합니다.

        Args:
            pdf_path (str): PDF 파일 경로

        Returns:
            str: 푸터가 제거된 전체 문서 텍스트
        """
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        cleaned_pages = [
            self.footer_pattern.sub('', p.page_content.strip()).strip()
            for p in pages
        ]
        return "\n".join(cleaned_pages)

    def split_articles(self, lines: List[str]) -> List[str]:
        """
        조문 제목('제X조(제목)') 단위로 조문을 분리합니다.

        Args:
            lines (List[str]): 라인 단위 텍스트 리스트

        Returns:
            List[str]: 조문별로 분리된 문자열 리스트
        """
        articles = []
        current_title = None
        current_content = []
        title_pattern = re.compile(r'^(제\d+조(?:의\d+)?\([^)]+\))')

        for line in map(str.strip, lines):
            if not line:
                continue
            match = title_pattern.match(line)
            if match:
                if current_title:
                    articles.append(f"{current_title}\n{''.join(current_content)}")
                current_title = match.group(1)
                rest = line[len(current_title):].strip()
                current_content = [rest] if rest else []
            elif current_title:
                current_content.append(line)
        if current_title:
            articles.append(f"{current_title}\n{''.join(current_content)}")
        return [a.strip() for a in articles]

    def parse_law_structure(self, law_text: str) -> dict:
        """
        문서 전체 텍스트를 장, 절, 조문 구조로 파싱합니다.

        Args:
            law_text (str): 정제된 법령 전체 텍스트

        Returns:
            dict: '서문', '장', '부칙' 키를 포함한 파싱 결과
        """
        lines = map(str.strip, law_text.replace('\r\n', '\n').replace('\r', '\n').split('\n'))
        parsed = {'서문': '', '장': {}, '부칙': ''}

        chapter_pattern = re.compile(r'^제\d+장\s+\S+')
        section_pattern = re.compile(r'^제\d+절\s+\S+')
        current_chapter, current_section = None, None
        chapter_buffer, buffer = {}, []
        preamble_done = False

        for i, line in enumerate(lines):
            if not line:
                continue
            if line.startswith("부칙"):
                if current_section:
                    chapter_buffer[current_section] = buffer
                elif buffer:
                    chapter_buffer['조문'] = buffer
                if current_chapter:
                    parsed['장'][current_chapter] = {
                        k: self.split_articles(v) for k, v in chapter_buffer.items()
                    }
                parsed['부칙'] = line
                break

            if chapter_pattern.match(line):
                if not preamble_done:
                    parsed['서문'] = '\n'.join(buffer).strip()
                    preamble_done = True
                elif current_chapter:
                    if current_section:
                        chapter_buffer[current_section] = buffer
                    elif buffer:
                        chapter_buffer['조문'] = buffer
                    parsed['장'][current_chapter] = {
                        k: self.split_articles(v) for k, v in chapter_buffer.items()
                    }
                current_chapter = line
                chapter_buffer, buffer = {}, []
                current_section = None
                continue

            if section_pattern.match(line):
                if current_section:
                    chapter_buffer[current_section] = buffer
                elif buffer:
                    chapter_buffer['조문'] = buffer
                current_section = line
                buffer = []
                continue

            buffer.append(line)

        if current_chapter:
            if current_section:
                chapter_buffer[current_section] = buffer
            elif buffer:
                chapter_buffer['조문'] = buffer
            parsed['장'][current_chapter] = {
                k: self.split_articles(v) for k, v in chapter_buffer.items()
            }

        return parsed

    def create_documents(self, pdf_path: str) -> List[Document]:
        """
        PDF 파일을 LangChain Document 리스트로 변환합니다.

        Args:
            pdf_path (str): 처리할 PDF 파일 경로

        Returns:
            List[Document]: 벡터 DB에 삽입 가능한 Document 객체 리스트
        """
        law_text = self.load_pdf(pdf_path)
        parsed_law = self.parse_law_structure(law_text)
        documents = []

        for chapter, sections in parsed_law["장"].items():
            for section, articles in sections.items():
                for article_text in articles:
                    title_line = article_text.split("\n", 1)[0]
                    metadata = {
                        "source": pdf_path,
                        "chapter": chapter,
                        "section": section,
                        "name": self.law_name,
                        "article": title_line
                    }
                    content = (
                        f"[법률정보]\n다음 조항은 {self.law_name} {chapter} {section}에서 발췌한 내용입니다.\n\n"
                        f"[법률조항]\n{article_text}"
                    )
                    documents.append(Document(page_content=content, metadata=metadata))
        return documents
