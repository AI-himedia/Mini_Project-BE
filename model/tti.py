from fastapi import APIRouter, Query
from fastapi.responses import Response, FileResponse
from PIL import Image
from io import BytesIO
import torch
from diffusers import AutoPipelineForText2Image
import uuid
import os
from .mini_lm import context_based_paragraph_split, load_embedding_model

tti_router = APIRouter()

OUTPUT_IMAGE_PATH = "./generated_images"

embedding_model, device = load_embedding_model()

def generate_image(result: str):
    device = torch.device("mps")
    
    pipe = AutoPipelineForText2Image.from_pretrained(
        "thibaud/sdxl_dpo_turbo", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    )
    pipe.to(device)

    images = pipe(result, num_inference_steps=6).images[0]

    buffer = BytesIO()
    images.save(buffer, format="JPEG")
    buffer.seek(0)

    return buffer.getvalue()

@tti_router.post("/sdxl_dpo_turbo")
def tti_view(text: str):
    paragraphs = context_based_paragraph_split(text, embedding_model)
    results = []
    
    for paragraph in paragraphs:
        image_data = generate_image(paragraph)
        
        # filename 생성
        filename = f"{uuid.uuid4()}.jpg"
        
        if not os.path.exists(OUTPUT_IMAGE_PATH):
            os.makedirs(OUTPUT_IMAGE_PATH)

        file_path = os.path.join(OUTPUT_IMAGE_PATH, filename)
        with open(file_path, "wb") as f:
            f.write(image_data)

        results.append({
            "filename": filename,
            "download_link": f"/sdxl_dpo_turbo/download/{filename}",
            "view_link": f"/sdxl_dpo_turbo/view/{filename}"
        })

    return results

@tti_router.get("/sdxl_dpo_turbo/view/{filename}")
def view_image(filename: str):
    file_path = os.path.join(OUTPUT_IMAGE_PATH, filename)
    return Response(content=open(file_path, "rb").read(), media_type="image/jpeg")

@tti_router.get("/sdxl_dpo_turbo/download/{filename}")
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
