import asyncio
import io
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set

import boto3
import fitz
import openai
import yaml
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from PIL import Image
from tqdm import tqdm

# === 로깅 설정 ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# === 설정 파일 로드 ===
def load_config() -> dict:
    """설정 파일을 로드합니다."""
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"설정 파일 로드 중 오류 발생: {e}")
        return {}


CONFIG = load_config()

# === 환경 세팅 ===
S3_BUCKET_NAME = CONFIG.get("s3", {}).get("bucket_name", "my-gpt-image-bucket")
S3_FOLDER_NAME = CONFIG.get("s3", {}).get("folder_name", "uploaded_images")
LOCAL_IMAGE_FOLDER = CONFIG.get("local", {}).get("image_folder", "./cropped_images")
DATA_DIR = CONFIG.get("local", {}).get("policy_dir", "./data/policy")
PERSIST_DIR = CONFIG.get("local", {}).get("persist_dir", "./chroma_db")
AWS_REGION = CONFIG.get("aws", {}).get("region", "ap-northeast-2")
CACHE_FILE = CONFIG.get("cache", {}).get("url_cache_file", "image_url_cache.json")
SUMMARY_DIR = CONFIG.get("output", {}).get("summary_dir", "summaries")
SUMMARY_CACHE_FILE = CONFIG.get("cache", {}).get(
    "summary_cache_file", "summary_cache.json"
)
MAX_IMAGE_SIZE = CONFIG.get("image", {}).get(
    "max_size", 1024
)  # 최대 이미지 크기 (픽셀)
PROGRESS_FILE = CONFIG.get("cache", {}).get("progress_file", "processing_progress.json")

# === S3 클라이언트 생성 ===
s3_client = boto3.client("s3", region_name=AWS_REGION)


def ensure_directories():
    """필요한 디렉토리들이 존재하는지 확인하고 생성합니다."""
    for directory in [DATA_DIR, LOCAL_IMAGE_FOLDER, SUMMARY_DIR, PERSIST_DIR]:
        os.makedirs(directory, exist_ok=True)


def optimize_image(image_path: str) -> Optional[bytes]:
    """이미지를 최적화합니다."""
    try:
        with Image.open(image_path) as img:
            # 이미지 크기 조정
            if max(img.size) > MAX_IMAGE_SIZE:
                ratio = MAX_IMAGE_SIZE / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # 이미지를 바이트로 변환
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=img.format or "JPEG", quality=85)
            return img_byte_arr.getvalue()
    except Exception as e:
        logger.error(f"이미지 최적화 중 오류 발생 ({image_path}): {e}")
        return None


def load_url_cache() -> Dict[str, str]:
    """캐시된 URL 정보를 로드합니다."""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"캐시 파일 로드 중 오류 발생: {e}")
    return {}


def save_url_cache(cache: Dict[str, str]):
    """URL 정보를 캐시 파일에 저장합니다."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"캐시 파일 저장 중 오류 발생: {e}")


async def upload_image_to_s3(
    image_path: str, url_cache: Dict[str, str]
) -> Optional[str]:
    """이미지를 S3에 업로드하고 URL을 반환합니다."""
    try:
        file_name = os.path.basename(image_path)

        # 캐시에 있는 경우 캐시된 URL 반환
        if file_name in url_cache:
            logger.info(f"캐시된 URL 사용: {file_name}")
            return url_cache[file_name]

        # 이미지 최적화
        optimized_image = optimize_image(image_path)
        if not optimized_image:
            return None

        # 새로운 이미지 업로드
        logger.info(f"새 이미지 업로드 중: {file_name}")
        s3_key = f"{S3_FOLDER_NAME}/{file_name}"

        # 비동기 업로드
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: s3_client.upload_fileobj(
                io.BytesIO(optimized_image), S3_BUCKET_NAME, s3_key
            ),
        )

        url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        url_cache[file_name] = url
        return url
    except Exception as e:
        logger.error(f"이미지 업로드 중 오류 발생 ({image_path}): {e}")
        return None


async def get_image_urls_from_folder(folder_path: str) -> List[str]:
    """폴더 내의 모든 이미지 URL을 가져옵니다."""
    url_cache = load_url_cache()
    image_urls = []

    image_files = [
        os.path.join(root, file_name)
        for root, _, files in os.walk(folder_path)
        for file_name in files
        if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
    ]

    for file_path in tqdm(image_files, desc="이미지 URL 수집 중"):
        url = await upload_image_to_s3(file_path, url_cache)
        if url:
            image_urls.append(url)

    save_url_cache(url_cache)
    return image_urls


def create_image_message_url(image_url: str) -> dict:
    """이미지 URL을 API 요청 형식에 맞게 변환합니다."""
    return {"type": "image_url", "image_url": {"url": image_url}}


def save_summary(summary: str, image_count: int, image_urls: List[str]):
    """요약 결과를 파일로 저장합니다."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"summary_{timestamp}_{image_count}images.txt"
    filepath = os.path.join(SUMMARY_DIR, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"요약 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"이미지 수: {image_count}\n")
            f.write("\n=== 이미지 URL 목록 ===\n")
            for url in image_urls:
                f.write(f"- {url}\n")
            f.write("\n=== 요약 내용 ===\n\n")
            f.write(summary)
        logger.info(f"요약 결과 저장 완료: {filepath}")
    except Exception as e:
        logger.error(f"요약 결과 저장 중 오류 발생: {e}")


def summarize_images(image_urls: List[str], custom_prompt: str = None) -> str:
    """TODO: 이미지들을 분석하고 요약합니다."""
    try:
        messages = [
            {
                "role": "system",
                "content": "당신은 고용노동부 카드뉴스를 정확하게 요약하는 전문 요약가입니다.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": custom_prompt
                        or "아래 이미지 링크들을 보고 핵심 내용을 통합해서 간단하게 요약해 주세요.",
                    },
                    *[create_image_message_url(url) for url in image_urls],
                ],
            },
        ]

        logger.info(f"GPT-4 Vision API 호출 중... (이미지 {len(image_urls)}개)")
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"이미지 요약 중 오류 발생: {e}")
        return f"요약 중 오류가 발생했습니다: {str(e)}"


def get_document(text: str, summary: str, pdf_dir: str) -> Document:
    """TODO:텍스트와 요약을 결합하여 Document 객체를 반환합니다."""
    document_format = f"""원본 텍스트 내용:
{text}

이미지 요약 내용:
{summary}

위 내용은 동일한 PDF 문서에서 추출된 것으로, 텍스트로 직접 추출된 내용과 이미지에서 추출된 내용을 포함합니다."""
    return Document(
        page_content=document_format,
        metadata={"source": "cardnews", "pdf_dir": pdf_dir},
    )


def preprocess_text(text: str) -> str:
    """TODO: 텍스트를 전처리합니다."""
    return text.replace("\n", " ").strip()


def load_progress() -> Tuple[Set[str], Set[str]]:
    """진행 상황을 로드합니다."""
    try:
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("processed_files", [])), set(
                    data.get("error_files", [])
                )
    except Exception as e:
        logger.error(f"진행 상황 로드 중 오류 발생: {e}")
    return set(), set()


def save_progress(processed_files: Set[str], error_files: Set[str]):
    """진행 상황을 저장합니다."""
    try:
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "processed_files": list(processed_files),
                    "error_files": list(error_files),
                    "last_updated": datetime.now().isoformat(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception as e:
        logger.error(f"진행 상황 저장 중 오류 발생: {e}")


async def process_pdf_file(file_path: str) -> Optional[Document]:
    """단일 PDF 파일을 처리합니다."""
    try:
        with fitz.open(file_path) as doc:
            text = " ".join([page.get_text() for page in doc])
            preprocessed_text = preprocess_text(text)

            pdf_name = os.path.splitext(os.path.basename(file_path))[0]
            cropped_images_dir = os.path.join(LOCAL_IMAGE_FOLDER, pdf_name)

            if not os.path.exists(cropped_images_dir):
                logger.warning(
                    f"이미지 디렉토리가 존재하지 않습니다: {cropped_images_dir}"
                )
                return None

            image_urls = await get_image_urls_from_folder(cropped_images_dir)
            if not image_urls:
                logger.warning(f"이미지 URL을 찾을 수 없습니다: {cropped_images_dir}")
                return None

            summary = summarize_images(image_urls)
            return get_document(preprocessed_text, summary, file_path)

    except Exception as e:
        logger.error(f"PDF 파일 처리 중 오류 발생 ({file_path}): {e}")
        return None


async def main():
    try:
        load_dotenv()
        ensure_directories()

        # 진행 상황 로드
        processed_files, error_files = load_progress()
        logger.info(
            f"이전 처리 내역: {len(processed_files)}개 처리됨, {len(error_files)}개 오류"
        )

        documents = []
        pdf_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(DATA_DIR)
            for file in files
            if file.lower().endswith(".pdf")
        ]

        # 처리할 파일 필터링
        files_to_process = [f for f in pdf_files if f not in processed_files]
        logger.info(f"처리할 파일 수: {len(files_to_process)}")

        for pdf_file in tqdm(files_to_process, desc="PDF 파일 처리 중"):
            try:
                if doc := await process_pdf_file(pdf_file):
                    documents.append(doc)
                    processed_files.add(pdf_file)
                    if pdf_file in error_files:
                        error_files.remove(pdf_file)
                else:
                    error_files.add(pdf_file)
            except Exception as e:
                logger.error(f"파일 처리 중 예외 발생 ({pdf_file}): {e}")
                error_files.add(pdf_file)

            # 진행 상황 저장
            save_progress(processed_files, error_files)

        if not documents:
            logger.warning("처리된 문서가 없습니다.")
            return

        logger.info(f"총 {len(documents)}개의 문서를 처리했습니다.")
        if error_files:
            logger.warning(f"처리 실패한 파일 수: {len(error_files)}")

        embeddings = OpenAIEmbeddings()
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="policy_docs",
            persist_directory=PERSIST_DIR,
        )
        logger.info("문서가 Chroma DB에 저장되었습니다.")

    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
