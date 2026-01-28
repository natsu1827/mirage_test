import torch
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
        base_output_dir='./__output/cls',

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
        weights='./__weights/MIRAGE-Base.pth',
        data_root='/path/to/data',
        data_set='dataset_name',
    )
    return args

def process_args(args):
    args.data_path = "./__image"
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
    args.resume = './__output/cls/v1/0/OCT_1_3class/mirage-base_finetune_w_7a2865b8/checkpoint-best-model.pth'
    misc.load_model(args=args, model=model, optimizer=optimizer)
    test_stats = evaluate_one(model, img_path, device)
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
    img_path = "/home/cji102_12/work/MIRAGE/__image/NORMAL_7_9.png"
    args = get_args()
    pred, conf = main(args, img_path)
    print("Prediction:", pred)
    print("Confidence:", conf)