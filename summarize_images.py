import os
from dotenv import load_dotenv
from typing import List, Dict
from PIL import Image
import base64
import json
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

class PolicyImageSummarizer:
    """
    PolicyImageSummarizer 클래스는 각 정책 폴더 안의 모든 이미지를 하나의 입력으로 처리하여
    해당 정책 전체를 요약한 구조화된 정보를 생성합니다.

    매개변수:
        image_root (str): 크롭된 이미지가 저장된 루트 디렉토리 경로
        output_path (str): 요약 결과를 저장할 JSON 파일 경로
    """
    def __init__(self, image_root: str, output_path: str):
        """
        클래스 초기화 함수. 이미지 디렉토리 경로와 결과 저장 경로를 설정하고,
        LLM 모델 및 출력 스키마, 프롬프트 템플릿을 초기화합니다.
        """
        load_dotenv()

        self.image_root = image_root
        self.output_path = output_path

        self.response_schemas = [
            ResponseSchema(name="Policy Name", description="Name of the policy or program"),
            ResponseSchema(name="Target Beneficiaries", description="Who the policy benefits"),
            ResponseSchema(name="Benefits", description="Details of provided benefits"),
            ResponseSchema(name="Application Method", description="How to apply or submit requests"),
            ResponseSchema(name="Others", description="Any additional relevant information")
        ]

        self.parser = StructuredOutputParser.from_response_schemas(self.response_schemas)
        self.format_instructions = self.parser.get_format_instructions()

        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

        self.prompt = ChatPromptTemplate.from_template(
            f"""
            You are given a collection of images from a policy announcement.  
            Below is a JSON array of objects, each containing a base64-encoded image URL.  

            Extract the following details in JSON format **exactly** matching this schema (no extra fields, no markdown):

            {self.format_instructions}

            Respond with ONLY the JSON object, no additional text.

            **INPUT**
            json:
            {{image_blocks}}
            """     
        )

    def encode_image_to_base64(self, image_path: str) -> str:
        """
        주어진 이미지 파일 경로를 base64 문자열로 인코딩하여 반환합니다.

        매개변수:
            image_path (str): 이미지 파일 경로

        반환:
            str: base64 인코딩된 이미지 문자열
        """
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def summarize_folder_images(self, image_paths: List[str]) -> Dict[str, str]:
        """
        하나의 정책 폴더 내의 여러 이미지를 LLM에 전달하여 통합 요약을 수행합니다.

        매개변수:
            image_paths (List[str]): 이미지 경로 리스트

        반환:
            Dict[str, str]: 정책 정보를 담은 구조화된 요약 결과
        """
        # 1) 각 이미지 파일을 base64로 인코딩하여 리스트로 모은다
        contents = []
        for image_path in image_paths:
            b64 = self.encode_image_to_base64(image_path)
            contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

        # 2) 파이썬 리스트 -> JSON 문자열로 변환
        payload = json.dumps(contents, ensure_ascii=False)

        # 3) JSON 문자열을 그대로 prompt에 넣어 모델에 전달
        messages = self.prompt.format_prompt(image_blocks=payload).to_messages()
        response = self.llm(messages)

        # 4) 모델 응답에서 JSON만 파싱
        raw = response.content.strip()
        try:
            return self.parser.parse(raw)
        except Exception:
            print(f"[JSON 파싱 실패] {image_paths}\nRaw response:\n{response.content}")
            raise

    def summarize_all_folders(self) -> Dict[str, Dict[str, str]]:
        all_results = {}
        for folder in os.listdir(self.image_root):
            folder_path = os.path.join(self.image_root, folder)
            if not os.path.isdir(folder_path):
                continue
            print(f"[📂 Processing folder]: {folder}")

            image_files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            if not image_files:
                continue

            try:
                summary = self.summarize_folder_images(image_files)
                all_results[folder] = summary
            except Exception as e:
                print(f"[⚠️ 폴더 요약 실패] {folder}: {e}")
                # 폴더 단위로 완전히 버리지 않고 넘어가거나,
                # all_results[folder] = {} 처럼 빈 값이라도 넣어줄 수 있습니다.

        return all_results


    def run(self):
        """
        전체 폴더 요약 작업을 실행하고 결과를 JSON 파일로 저장합니다.
        """
        results = self.summarize_all_folders()
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[✅ Summaries written to]: {self.output_path}")
