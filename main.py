from fastapi import FastAPI, File, UploadFile, Response

from diffusers import AutoPipelineForText2Image
from io import BytesIO
import torch

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

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