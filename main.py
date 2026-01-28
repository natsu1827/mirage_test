from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from pathlib import Path
import requests
from run_cls_infer import cls_oct
from run_seg_infer import seg_oct

app = FastAPI(title="AI Inference Service")

class InferRequest(BaseModel):
    image_url: str

class InferResponse(BaseModel):
    diagnosis: str
    score: float
    seg_path: str

@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    image_url = req.image_url
    # 模擬：下載影像 + 推論
    await asyncio.sleep(0.3)

    # TODO: 這裡換成你的模型推論
    pred, conf = cls_oct(image_url)
    save_path = seg_oct(image_url)

    return InferResponse(diagnosis=pred, score=conf, seg_path=str(save_path))
