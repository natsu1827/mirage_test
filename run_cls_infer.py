import torch
import sys
import os
import io
import tempfile
from mutils.gcs_utils import download_file_from_gcs, download_bytes_from_gcs, list_gcs_files
from mutils import misc
from mutils.classification_one import evaluate_one
from mutils.misc import fix_seeds
from fm_cls_config import fm_config_factory
from argparse import Namespace
from downloadimage import download

def get_args():
    args = Namespace(
        # Model parameters
        input_size=512,
        drop_path=0.1,
        weight_decay=0.05,

        # Optimizer parameters
        lr=1e-5,
        layer_decay=0.75,
        min_lr=1e-8,
        warmup_epochs=10,
        smoothing=0.1,
        accum_iter=1,

        # Supervised training params
        linear_probing=False,
        resume='',
        pool='global',
        base_output_dir='gs://oct-image-output-test-v1',

        # Data parameters
        num_workers=8,
        pin_mem=True,

        # Training parameters
        device='cuda',
        seed=0,
        start_epoch=0,
        batch_size=16,
        epochs=1000,
        eval=False,
        early_stopping_epochs=20,
        early_stopping_delta=0.001,
        early_stopping_delta_two=0.001,
        early_start_from=20,
        dry_run=False,
        version='v1',
        overwrite=False,
        val_metric='bacc',
        val_metric_two='loss',
        save_predictions=False,
        fill=None,
        affine=True,

        # Required arguments
        weights='gs://oct-mirage-model-test-v1/MIRAGE-Base.pth',
        data_root='/path/to/data',
        data_set='dataset_name',
    )
    return args

def download_bytes_from_gcs(gcs_uri: str) -> bytes:
    """解析 GCS URI 並下載為原始 Bytes"""
    try:
        storage_client = storage.Client()
        clean_uri = gcs_uri.replace("gs://", "")
        if "/" not in clean_uri:
            raise ValueError(f"Invalid GCS URI format: {gcs_uri}")
        bucket_name, blob_name = clean_uri.split("/", 1)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        print(f"⬇️ Downloading from GCS: {gcs_uri}")
        return blob.download_as_bytes()
    except Exception as e:
        raise RuntimeError(f"Failed to download {gcs_uri}: {e}")

def process_args(args):
    args.data_path = os.getenv("INPUT_BUCKET_URI", "gs://oct-image-test-v1")
    num_classes = 3
    args.num_classes = num_classes
    args.batch_size = 1
    return args

def main(args, img_path):
    fix_seeds(args.seed)
    device = torch.device(args.device)
    args = process_args(args)

    model_config = None
    # model_name = None
    for kw in fm_config_factory.keys():
        if kw in args.weights.lower():
            model_config = fm_config_factory[kw](args)
            # model_name = kw
            break
    if model_config is None:
        raise ValueError(f"Unknown model: {args.weights}")

    # Initialize the model
    model = model_config.model
    args = model_config.args
    model_config.set_requires_grad()
    model.to(device)

    optimizer = model_config.get_optimizer(model)

    # Evaluate on the best checkpoint
    args.resume = 'gs://oct-mirage-model-test-v1/cls/checkpoint-best-model.pth'
    misc.load_model(args=args, model=model, optimizer=optimizer)

    # 處理 GCS 圖片路徑
    if str(img_path).startswith("gs://"):
        img_bytes = download_bytes_from_gcs(str(img_path))
        img_input = io.BytesIO(img_bytes)
    else:
        img_input = img_path

    test_stats = evaluate_one(model, img_input, device)
    labels = ['AMD', 'DME', 'Normal']
    assert test_stats is not None
    pred_idx = test_stats['Prediction'].item()
    confidence = test_stats['Confidence'].max().item()

    return labels[pred_idx], confidence

def cls_oct(image_url):
    args = get_args()
    img_path = download(image_url)
    pred, conf = main(args, img_path)
    return pred, conf

if __name__ == '__main__':
    args = get_args()
    
    # 1. 決定輸入圖片列表 (自動抓取或單一指定)
    img_list = []
    
    if len(sys.argv) > 1:
        img_list = [sys.argv[1]]
    else:
        input_path = str(args.data_path)
        print(f"🔍 自動抓取圖片路徑: {input_path}")
        
        if input_path.startswith("gs://"):
            img_list = list_gcs_files(input_path)
        else:
            if os.path.exists(input_path):
                img_list = sorted([str(os.path.join(input_path, f)) for f in os.listdir(input_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    if not img_list:
        print(f"❌ 找不到可處理的圖片。請確認路徑: {args.data_path}")
        sys.exit(0)

    # 2. 批次執行 Inference
    print(f"🚀 發現 {len(img_list)} 張圖片，開始進行推論...")
    for img_path in img_list:
        try:
            pred, conf = main(args, img_path)
            print(f"✅ 完成: {os.path.basename(img_path)}")
            print(f"   -> 預測結果: {pred} (信心度: {conf:.4f})")
        except Exception as e:
            print(f"❌ 發生錯誤於 {img_path}: {e}")
            import traceback
            traceback.print_exc()