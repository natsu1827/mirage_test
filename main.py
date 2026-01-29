from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import logging
import uuid
import os
from mutils.gcs_utils import upload_file_to_gcs

# Import Controller
from oct_contronller import process_oct_image

# Initialize FastAPI
app = FastAPI(title="OCT AI Inference Service")

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GCS Input Bucket for Upload testing
DEFAULT_INPUT_BUCKET = os.getenv("INPUT_BUCKET_NAME", "test-oct-image")

# --- Pydantic Models: Matching CFP Style ---
class PredictionRequest(BaseModel):
    image_gcs_path: str = Field(..., description="GCS Path (gs://...) or HTTP URL")
    request_id: Optional[str] = None

class PredictionResponse(BaseModel):
    status: str
    request_id: str
    results: dict

# --- 1. Health Check & Portal ---
@app.get("/", response_class=HTMLResponse)
async def health_check_portal():
    """Health Check and Premium User Interface loaded from file"""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        logger.error(f"Failed to load template: {e}")
        return HTMLResponse(content="<h1>OCT AI Service - Template Error</h1>", status_code=500)

# --- 2. Inference Entry Point: Aligned with CFP ---
@app.post("/predict/oct")
async def predict_oct_endpoint(req: PredictionRequest):
    """
    Main entry point for OCT inference.
    Routes to oct_contronller.process_oct_image.
    """
    request_id = req.request_id or str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Input Received: {req.image_gcs_path}")
    
    try:
        # Call the Controller
        result = await process_oct_image(req.image_gcs_path, request_id)
        
        # Ensure request_id is in response
        result["request_id"] = request_id
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"[{request_id}] Prediction Failure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 3. Upload Helper for Testing ---
@app.post("/upload")
async def upload_test_image(file: UploadFile = File(...)):
    """Helper to upload local images to GCS for testing"""
    try:
        unique_filename = f"web_{uuid.uuid4().hex[:8]}_{file.filename}"
        gcs_uri = f"gs://{DEFAULT_INPUT_BUCKET}/{unique_filename}"
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            upload_file_to_gcs(tmp_path, gcs_uri)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        return {"status": "success", "gcs_uri": gcs_uri}
    except Exception as e:
        logger.error(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    import tempfile
    # Start on port from env (Cloud Run) or 8080
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
