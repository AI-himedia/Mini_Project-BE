from fastapi import FastAPI, File, UploadFile, Response

from diffusers import AutoPipelineForText2Image
from io import BytesIO
import torch

import model.ocr as ocr

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/img_ocr")
async def img_ocr(
    image: UploadFile = File(...)
):
    contents = await image.read()
    # filename = f"temp_{image.filename}"
    # with open(filename, "wb") as f:
    #     f.write(contents)
    return Response(ocr.extract_text_from_image(contents))

@app.post("/sdxl_dpo_turbo")
def tti(input_text: str):
    device = torch.device("mps")

    pipe = AutoPipelineForText2Image.from_pretrained(
    "thibaud/sdxl_dpo_turbo", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    )
    pipe.to(device)

    images = pipe(input_text, num_inference_steps=6).images[0]

    # case1 이미지 view
    buffer = BytesIO()
    images.save(buffer, format="JPEG")
    buffer.seek(0)

    return Response(buffer.getvalue(), media_type="image/jpeg")

    # case2 이미지 다운로드 링크
    # images.save(OUTPUT_IMAGE_PATH)

    # return FileResponse(OUTPUT_IMAGE_PATH, media_type="image/jpg", filename="generated_image.jpg")

# @app.post("/img_ocr")
# async def img_ocr(
#     image: UploadFile = File(...)
# ):
#     contents = await image.read()
#     filename = f"temp_{image.filename}"
#     with open(filename, "wb") as f:
#         f.write(contents)
    
#     # STEP 3: Load the input image.
#     image = mp.Image.create_from_file(filename)
#     # STEP 4: Detect objects in the input image.
#     detection_result = detector.detect(image)
#     # print(detection_result)
#     # STEP 5: Process the detection result. In this case, visualize it.
#     objects = []
#     for detection in detection_result.detections:
#         objects.append(detection)
#     print(f"Find Object : {len(objects)}")
        
#     image_copy = np.copy(image.numpy_view())
#     annotated_image = visualize(image_copy, detection_result)
#     rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
#     cv2.imwrite("test.jpg", rgb_annotated_image)
#     return FileResponse("test.jpg")