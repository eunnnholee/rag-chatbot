import re


def split_articles(lines):
    """
    조문 본문에서 개행으로 인한 단어 분리를 방지하기 위해
    줄 간 공백 없이 연결한다. 조문은 '제X조(제목)' 형식으로 시작한다.
    """
    # 조문을 저장할 리스트
    articles = []
    # 현재 처리 중인 조문의 제목
    current_title = None
    # 현재 처리 중인 조문의 내용
    current_content = []

    # '제X조(제목)' 형식을 찾기 위한 정규표현식 패턴
    title_pattern = re.compile(r"^(제\d+조(?:의\d+)?\([^)]+\))")

    # 각 줄을 순회하며 처리
    for line in map(str.strip, lines):
        if not line:  # 빈 줄 건너뛰기
            continue

        # 조문 제목 패턴이 매칭되는지 확인
        match = title_pattern.match(line)
        if match:
            # 이전 조문이 있으면 저장
            if current_title:
                articles.append(f"{current_title}\n{''.join(current_content)}")
            # 새로운 조문 시작
            current_title = match.group(1)
            rest = line[len(current_title) :].strip()
            current_content = [rest] if rest else []
        elif current_title:
            # 조문 내용 추가
            current_content.append(line)

    # 마지막 조문 저장
    if current_title:
        articles.append(f"{current_title}\n{''.join(current_content)}")

    return [a.strip() for a in articles]


def parse_law_with_sections(law_text):
    """
    전체 법령 텍스트를 서문, 장, 절, 조문 단위로 구조화하여 파싱한다.
    """
    # 줄바꿈 문자 정규화
    law_text = law_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = map(str.strip, law_text.split("\n"))

    # 파싱 결과를 저장할 딕셔너리
    parsed_law = {"서문": "", "장": {}, "부칙": ""}

    # 장과 절을 찾기 위한 정규표현식 패턴
    chapter_pattern = re.compile(r"^제\d+장\s+\S+")
    section_pattern = re.compile(r"^제\d+절\s+\S+")

    # 현재 처리 중인 위치 추적용 변수들
    current_chapter = None
    current_section = None
    chapter_buffer = {}
    buffer = []
    preamble_done = False

    # 각 줄을 순회하며 처리
    for i, line in enumerate(lines):
        if not line:
            continue

        # 부칙 처리
        if line.startswith("부칙"):
            if current_section:
                chapter_buffer[current_section] = buffer
            elif buffer:
                chapter_buffer["조문"] = buffer
            if current_chapter:
                parsed_law["장"][current_chapter] = {
                    k: split_articles(v) for k, v in chapter_buffer.items()
                }
            parsed_law["부칙"] = "\n".join([line] + list(lines)[i + 1 :]).strip()
            break

        # 장 처리
        if chapter_pattern.match(line):
            if not preamble_done:
                parsed_law["서문"] = "\n".join(buffer).strip()
                preamble_done = True
            elif current_chapter:
                if current_section:
                    chapter_buffer[current_section] = buffer
                elif buffer:
                    chapter_buffer["조문"] = buffer
                parsed_law["장"][current_chapter] = {
                    k: split_articles(v) for k, v in chapter_buffer.items()
                }
            current_chapter = line
            chapter_buffer = {}
            current_section = None
            buffer = []
            continue

        # 절 처리
        if section_pattern.match(line):
            if current_section:
                chapter_buffer[current_section] = buffer
            elif buffer:
                chapter_buffer["조문"] = buffer
            current_section = line
            buffer = []
            continue

        buffer.append(line)

    # 마지막 장 저장
    if current_chapter:
        if current_section:
            chapter_buffer[current_section] = buffer
        elif buffer:
            chapter_buffer["조문"] = buffer
        parsed_law["장"][current_chapter] = {
            k: split_articles(v) for k, v in chapter_buffer.items()
        }

    return parsed_law
