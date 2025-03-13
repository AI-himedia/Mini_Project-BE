from fastapi import FastAPI, File, UploadFile, Response
from diffusers import AutoPipelineForText2Image
from io import BytesIO
from model import routers
import torch

app = FastAPI()

for router in routers:
    app.include_router(router)

@app.get("/")
def read_root():
    return {"Hello": "World"}
