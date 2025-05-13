# STEP 1: PDF에서 이미지 블록만 크롭하여 폴더별 저장
from pdf_image_cropper import PDFImageCropper

cropper = PDFImageCropper(input_folder="./data/policy")
cropper.run()  # 여러 PDF 자동 순회하며 이미지 추출

# STEP 2: 크롭된 이미지들을 요약
from summarize_images import PolicyImageSummarizer

summarizer = PolicyImageSummarizer(
    image_root="./cropped_images",
    output_path="structured_image_summaries.json"
)
summarizer.run()
