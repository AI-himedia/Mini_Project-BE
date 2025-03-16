# model/tti.py

from fastapi import APIRouter, Query
from fastapi.responses import Response, FileResponse
from PIL import Image
from io import BytesIO
import torch
from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline
import uuid
import os
import model.translation as translation
import time

tti_router = APIRouter()

OUTPUT_IMAGE_PATH = "./generated_images"

diary_text = translation.translated_diary
model_name = "etri-vilab/koala-700m-llava-cap"

# 사용 가능한 모델 리스트
AVAILABLE_MODELS = {
    "thibaud/sdxl_dpo_turbo": {
        "model_name": "thibaud/sdxl_dpo_turbo",
        "pipeline": AutoPipelineForText2Image,
        "num_inference_steps": 6
    },
    "etri-vilab/koala-700m-llava-cap": {
        "model_name": "etri-vilab/koala-700m-llava-cap",
        "pipeline": StableDiffusionXLPipeline,
        "num_inference_steps": 35
    },
    "sd-community/sdxl-flash": {
        "model_name": "sd-community/sdxl-flash",
        "pipeline": StableDiffusionXLPipeline,
        "num_inference_steps": 7
    }
}

# 텍스트 기반 이미지 생성, 바이너리 데이트를 반환
def generate_image(result: str, model_name: str):
    # 기기 자동 선택
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")
    
    if model_name == "thibaud/sdxl_dpo_turbo":
        kwargs = {
            "pretrained_model_or_path": AVAILABLE_MODELS[model_name]["model_name"],
            "torch_dtype": torch.float16,
            "use_safetensors": True,
            "variant": "fp16"
        }
    else:
        kwargs = {
            "pretrained_model_name_or_path": AVAILABLE_MODELS[model_name]["model_name"],
            "torch_dtype": torch.float16
        }
    
    pipe = AVAILABLE_MODELS[model_name]["pipeline"].from_pretrained(**kwargs)
    pipe.to(device)

    negative_prompt = (
                        """poorly drawn hands and feet,
                        extra fingers, fused limbs,
                        distorted facial features,
                        cropped,
                        distorted anatomy,
                        poorly drawn face,
                        bad anatomy"""
                        )

    images = pipe(result, num_inference_steps=AVAILABLE_MODELS[model_name]["num_inference_steps"], negative_prompt=negative_prompt).images[0]
    # images = pipe(result, num_inference_steps=AVAILABLE_MODELS[model_name]["num_inference_steps"]).images[0]

    buffer = BytesIO()
    images.save(buffer, format="JPEG")
    buffer.seek(0)

    return buffer.getvalue()

# 영어문단 3개에 대한 이미지 정보 반환
@tti_router.post("/image/generate")
def tti_view(model_name: str):
    start_time = time.time()
    diary_text = translation.translated_diary

    if not diary_text:
        return {"error": "No translated text available"}
    
    results = []
    
    for text in diary_text:
        image_data = generate_image(text, model_name)
        
        # filename 랜덤 생성
        filename = f"{uuid.uuid4()}.jpg"
        
        if not os.path.exists(OUTPUT_IMAGE_PATH):
            os.makedirs(OUTPUT_IMAGE_PATH)

        file_path = os.path.join(OUTPUT_IMAGE_PATH, filename)
        with open(file_path, "wb") as f:
            f.write(image_data)

        results.append({
            "filename": filename,
            "download_link": f"/image/download/{filename}",
            "view_link": f"/image/view/{filename}"
        })

        end_time = time.time()
        elapsed_time = end_time - start_time

    print(f"이미지 출력 시간: {elapsed_time:.4f} 초")

    return {"message": "Images generated successfully", "images": results}

# 저장된 이미지 조회 API
@tti_router.get("/image/view/{filename}")
def view_image(filename: str):
    file_path = os.path.join(OUTPUT_IMAGE_PATH, filename)
    return Response(content=open(file_path, "rb").read(), media_type="image/jpeg")

# 생성된 이미지 다운로드 API
@tti_router.get("/image/download/{filename}")
def download_image(filename: str, new_filename: str = Query(None)):
    file_path = os.path.join(OUTPUT_IMAGE_PATH, filename)
    
    if new_filename:
        # Content-Disposition 헤더 설정
        headers = {
            "Content-Disposition": "attachment; filename*=UTF-8''{new_filename}".format(new_filename=new_filename)
        }
    else:
        headers = {
            "Content-Disposition": "attachment; filename*=UTF-8''{filename}".format(filename=filename)
        }
    
    return FileResponse(file_path, media_type="image/jpg", headers=headers)

tti_view(model_name)