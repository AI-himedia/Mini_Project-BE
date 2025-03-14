from fastapi import APIRouter, Query, File, UploadFile
from fastapi.responses import Response, FileResponse
import easyocr
import time

ocr_router = APIRouter()

def extract_text_from_image(image_path):
    # Initialize the OCR model
    reader = easyocr.Reader(['ko', 'en'])  # Load the model into memory

    # Read text from the image
    extracted_text_list = reader.readtext(image_path, detail=0)

    # Restructure extracted text into a single string
    extracted_text = ' '.join(extracted_text_list)

    return extracted_text

# Example usage
if __name__ == "__main__":
    image_path = 'handwriting_diary_log.png'
    extracted_text = extract_text_from_image(image_path)
    print('Extracted Text:', extracted_text)



@ocr_router.post("/img_ocr")
async def img_ocr(
    image: UploadFile = File(...)
):
    contents = await image.read()
    # filename = f"temp_{image.filename}"
    # with open(filename, "wb") as f:
    #     f.write(contents)
    return Response(extract_text_from_image(contents))
# Measure the time taken for OCR
# start_time = time.time()
# 이미지 경로 지정
# extracted_text_list = reader.readtext('handwriting_diary_log.png', detail = 0)
# Restructure extracted text into a single string
# extracted_text = ' '.join(extracted_text_list)
# end_time = time.time()

# Calculate the time taken
# time_taken = end_time - start_time

# Calculate similarity
# similarity_rate = calculate_similarity(actual_text_handwriting_diary_log, extracted_text)



# Print results
print('-------------------------------------------------------------------------------------')
# print('손글씨 일기 캡쳐본 OCR 테스트 결과')
# print('Extracted Text:', extracted_text)
# print('Time Taken (seconds):', time_taken)
# print('Similarity Rate:', similarity_rate)