import fitz  # PyMuPDF
import os
from typing import List
from PIL import Image

class PDFImageCropper:
    """
    PDFImageCropper 클래스는 PDF 문서에서 이미지 또는 표로 추정되는 시각적 블록을 탐지하여
    해당 영역만 크롭한 이미지를 저장하는 기능을 수행합니다.

    전체 문서 기준 첫 번째 이미지는 생략하며, 각 PDF 파일은 별도의 폴더로 구분되어 저장됩니다.

    매개변수:
        input_folder (str): PDF 파일들이 저장된 입력 폴더 경로
        output_root (str): 크롭된 이미지를 저장할 루트 디렉토리 경로 (기본값: './cropped_images')
    """
    def __init__(self, input_folder: str, output_root: str = "./cropped_images"):
        self.input_folder = input_folder
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)

    def extract_images_only(self, file_path: str) -> List[str]:
        """
        단일 PDF 파일에서 이미지 블록만 추출하여 크롭된 이미지를 저장합니다.
        - 첫 번째 이미지 블록은 무시됩니다.
        - 저장 디렉토리는 PDF 파일명을 기준으로 폴더가 생성됩니다.

        매개변수:
            file_path (str): 처리할 PDF 파일의 전체 경로

        반환값:
            List[str]: 저장된 이미지 파일들의 경로 리스트
        """
        doc = fitz.open(file_path)
        image_paths = []

        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        output_image_dir = os.path.join(self.output_root, base_filename)
        os.makedirs(output_image_dir, exist_ok=True)

        skipped_first_image = False

        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            for i, block in enumerate(blocks):
                print(f"  - Block {i}: type={block['type']}, bbox={block['bbox']}")
                if block["type"] == 1:  # 이미지 블록
                    if not skipped_first_image:
                        skipped_first_image = True
                        print("---첫 번째 이미지 블록 생략---")
                        continue

                    rect = fitz.Rect(block["bbox"])
                    pix = page.get_pixmap(clip=rect)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    img_path = os.path.join(output_image_dir, f"page{page_num+1}_crop{i}.png")
                    img.save(img_path)
                    image_paths.append(img_path)

        return image_paths

    def run(self):
        """
        입력 폴더 내 모든 PDF 파일에 대해 이미지 크롭을 수행합니다.
        각 파일마다 extract_images_only()를 호출하며, 처리 결과를 출력합니다.
        """
        for filename in os.listdir(self.input_folder):
            if filename.endswith(".pdf"):
                file_path = os.path.join(self.input_folder, filename)
                print(f"\n[🔍 처리 중] {filename}")
                cropped_images = self.extract_images_only(file_path)
                print(f"[✅ {filename}] Crop된 이미지 개수: {len(cropped_images)}")
                for img_path in cropped_images:
                    print(f"- {img_path}")
