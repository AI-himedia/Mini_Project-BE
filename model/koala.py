from fastapi import FastAPI, UploadFile, File, Response, APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
from io import BytesIO
import base64


# FastAPI 애플리케이션 설정
router = APIRouter()

device = "cuda"
# device = "mps"

# Stable Diffusion XL 파이프라인 로드
pipe = StableDiffusionXLPipeline.from_pretrained("etri-vilab/koala-700m-llava-cap", torch_dtype=torch.float16)
pipe.to(device)


@router.post("/preview_image")
async def preview_image(prompt: str):
    image = pipe(prompt=prompt, num_inference_steps=30).images[0]

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    buffered.seek(0)
    # 다운로드 가능한 형태로 반환 -> return 변경
    # headers = {
    #     "Content-Disposition": "attachment; filename=preview_image.jpg"
    # }

    return Response(buffered.getvalue(), media_type="image/jpeg")