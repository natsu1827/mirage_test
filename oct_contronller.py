import uuid
import os
import io
import json
import logging
import tempfile
from pathlib import Path

# Import inference modules
import run_cls_infer
import run_seg_infer
import amd_analysis
from mutils.gcs_utils import download_bytes_from_gcs, upload_file_to_gcs, list_gcs_files
from amd_analysis import AMDVisualizer, is_gcs

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GCS Output Bucket from Env
OUTPUT_BUCKET = os.getenv("OUTPUT_BUCKET_NAME", "test-oct-image-output")

async def process_oct_image(image_gcs_path: str, request_id: str = None):
    """
    Orchestrates the OCT image analysis flow:
    1. Classification (run_cls_infer)
    2. Segmentation (run_seg_infer)
    3. AMD Analysis (amd_analysis)
    """
    if not request_id:
        request_id = str(uuid.uuid4())[:8]
    
    logger.info(f"[{request_id}] 🚀 Starting OCT Processing for {image_gcs_path}")
    
    filename = os.path.basename(image_gcs_path)
    # Output directory on GCS for this request
    gcs_output_dir = f"gs://{OUTPUT_BUCKET}/{request_id}"
    
    try:
        # --- 1. Classification (模型分類) ---
        logger.info(f"[{request_id}] 🧠 Running Classification...")
        cls_args = run_cls_infer.get_args()
        # Ensure we use GCS for model weights if not set
        pred_label, confidence = run_cls_infer.main(cls_args, image_gcs_path)
        logger.info(f"[{request_id}] Classification Result: {pred_label} ({confidence:.4f})")

        # --- 2. Segmentation (模型分割) ---
        logger.info(f"[{request_id}] 🖼️ Running Segmentation...")
        seg_args = run_seg_infer.get_args()
        # Set segmentation output to GCS
        seg_args.output_dir = gcs_output_dir
        mask_path = run_seg_infer.main(seg_args, image_gcs_path)
        logger.info(f"[{request_id}] Segmentation Mask saved to: {mask_path}")

        # --- 3. AMD Analysis (量化分析與透明標註) ---
        logger.info(f"[{request_id}] 📊 Running AMD Analysis...")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_raw = os.path.join(tmp_dir, filename)
            local_mask = os.path.join(tmp_dir, f"mask_{filename}")
            
            # Download Raw & Mask for PIL processing
            raw_bytes = download_bytes_from_gcs(image_gcs_path)
            with open(local_raw, "wb") as f:
                f.write(raw_bytes)
                
            mask_bytes = download_bytes_from_gcs(str(mask_path))
            with open(local_mask, "wb") as f:
                f.write(mask_bytes)
            
            # Set global OUTPUT_DIR in amd_analysis for upload
            # We use a context-safe approach if it was a class, but here we set the global
            old_output_dir = amd_analysis.OUTPUT_DIR
            amd_analysis.OUTPUT_DIR = gcs_output_dir
            
            viz = AMDVisualizer(local_raw, local_mask, filename, pixel_spacing=0.003)
            viz.draw_contours()
            viz.draw_vertical_caliper()
            viz.draw_horizontal_caliper()
            
            # This triggers upload to GCS because we set amd_analysis.OUTPUT_DIR
            final_img_path, overlay_filename, json_filename = viz.save_results()
            
            # Restore original global
            amd_analysis.OUTPUT_DIR = old_output_dir
            
            # Read back analysis JSON to include in response
            with open(os.path.join(tmp_dir, json_filename), "r", encoding="utf-8") as f:
                analysis_data = json.load(f)

            # --- 4. Final Response Construction ---
            # Define color based on diagnosis
            color_map = {
                "Normal": "#00FF00", # Green
                "AMD": "#FF0000",    # Red
                "DME": "#FFA500"     # Orange
            }
            
            response = {
                "status": "success",
                "request_id": request_id,
                "results": {
                    "raw_image": image_gcs_path,
                    "overlay_image": f"{gcs_output_dir}/{overlay_filename}",
                    "analyzed_image": f"{gcs_output_dir}/{os.path.basename(final_img_path)}",
                    "diagnosis": pred_label,
                    "confidence": float(confidence),
                    "analysis_data": analysis_data,
                    "annotation": {
                        "type": "Diagnosis",
                        "data": pred_label,
                        "color": color_map.get(pred_label, "#FFFFFF")
                    }
                }
            }
            
            logger.info(f"[{request_id}] ✅ OCT Processing Completed.")
            return response

    except Exception as e:
        logger.error(f"[{request_id}] ❌ Error during processing: {e}")
        return {
            "status": "error",
            "request_id": request_id,
            "message": str(e)
        }

if __name__ == "__main__":
    # Test execution
    import asyncio
    test_gcs_path = "gs://test-oct-image/NORMAL_7_9.png"
    asyncio.run(process_oct_image(test_gcs_path, "test_req_001"))
