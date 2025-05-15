import os
import boto3
import openai
import json
import logging
import asyncio
import yaml
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
from PIL import Image
import io

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
AWS_REGION = CONFIG.get("aws", {}).get("region", "ap-northeast-2")
CACHE_FILE = CONFIG.get("cache", {}).get("url_cache_file", "image_url_cache.json")
SUMMARY_DIR = CONFIG.get("output", {}).get("summary_dir", "summaries")
SUMMARY_CACHE_FILE = CONFIG.get("cache", {}).get(
    "summary_cache_file", "summary_cache.json"
)
MAX_IMAGE_SIZE = CONFIG.get("image", {}).get(
    "max_size", 1024
)  # 최대 이미지 크기 (픽셀)
BATCH_SIZE = CONFIG.get("api", {}).get("batch_size", 5)  # API 호출 배치 크기

# === S3 클라이언트 생성 ===
s3_client = boto3.client("s3", region_name=AWS_REGION)


def ensure_directories():
    """필요한 디렉토리들이 존재하는지 확인하고 생성합니다."""
    os.makedirs(SUMMARY_DIR, exist_ok=True)


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

    # 이미지 파일 목록 수집
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                image_files.append(os.path.join(root, file_name))

    # 진행 상황 표시와 함께 URL 수집
    for file_path in tqdm(image_files, desc="이미지 URL 수집 중"):
        url = await upload_image_to_s3(file_path, url_cache)
        if url:
            image_urls.append(url)

    # 캐시 저장
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
    """이미지들을 분석하고 요약합니다."""
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


async def main():
    try:
        load_dotenv()
        ensure_directories()

        logger.info("이미지 URL 수집 시작")
        image_urls = await get_image_urls_from_folder(LOCAL_IMAGE_FOLDER)
        logger.info(f"총 {len(image_urls)}개의 이미지 URL 수집 완료")

        if not image_urls:
            logger.warning("처리할 이미지가 없습니다.")
            return

        # 처음 5개 이미지만 처리
        target_urls = image_urls[:5]
        logger.info(f"처리할 이미지 수: {len(target_urls)}")

        summary = summarize_images(target_urls)
        logger.info("요약 완료")

        print("\n✅ 최종 요약 결과:\n")
        print(summary)

        # 요약 결과 저장
        save_summary(summary, len(target_urls), target_urls)

    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
