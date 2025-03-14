from fastapi import FastAPI
from model import routers
import uvicorn

app = FastAPI()

for router in routers:
    app.include_router(router)

@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)