import argparse
import json
import warnings
from functools import partial
from pathlib import Path
from typing import Dict

from PIL import Image
import numpy as np
from skimage import io
import torch
from torch.amp.autocast_mode import autocast
import yaml

from mirage.model import model_factory
from mirage.output_adapters import (
    ConvNeXtAdapter,
    DPTOutputAdapter,
    LinearSegAdapter,
    SegmenterMaskTransformerAdapter
)
import mutils.checkpoint
from mutils.native_scaler import NativeScalerWithGradNormCount as NativeScaler
from mutils.seg_one_data import simple_transform
from mutils.optim_factory import LayerDecayValueAssigner, create_optimizer
from mutils.gdice import CEGDiceLoss
from mutils.misc import fix_seeds, SortingHelpFormatter
from fm_seg_config import fm_factory
from downloadimage import download
from argparse import Namespace

def get_args():
    args = Namespace(
        # ===== Training =====
        batch_size=8,
        epochs=200,
        save_ckpt_freq=20,

        # ===== Task =====
        in_domains='bscan',
        mapping_fn=None,

        # ===== Model =====
        model='multivit_base',
        num_global_tokens=1,
        patch_size=32,
        input_size=1024,
        drop_path_encoder=0.1,
        learnable_pos_emb=False,
        freeze_encoder=True,
        ignore_index=-1,
        output_adapter='convnext',
        decoder_interpolate_mode='bilinear',
        decoder_main_tasks='bscan',
        loss='CE-ignore-bg',
        minmax=False,

        # ===== Optimizer =====
        opt='adamw',
        opt_eps=1e-8,
        opt_betas=[0.9, 0.999],
        clip_grad=None,
        momentum=0.9,
        weight_decay=0.05,
        weight_decay_end=None,
        decoder_decay=None,
        no_lr_scale_list='',
        lr=1e-4,
        warmup_lr=1e-6,
        min_lr=0.0,
        layer_decay=0.75,
        warmup_epochs=1,
        warmup_steps=-1,

        # ===== Dataset =====
        test_data_path=None,
        eval_freq=1,

        # ===== Runtime =====
        base_output_dir='./__output/seg',
        device='cuda',
        seed=42,
        resume='',
        auto_resume=True,
        save_ckpt=True,
        start_epoch=0,
        infer_only=True,
        test=True,
        num_workers=8,
        pin_mem=True,
        fp16=True,

        # ===== Logging =====
        log_images_freq=5,
        log_images=False,
        show_user_warnings=False,
        version='v1',

        # ===== Required =====
        weights="__weights/MIRAGE-Base.pth",
        data_path="__datasets/Segmentation/AROI",
    )

    return process_args(args)


def process_args(args):
    args.in_domains = args.in_domains.split('-')
    domains = args.in_domains
    if isinstance(args.patch_size, int):
        args.patch_size = {d: (args.patch_size, args.patch_size) for d in domains}

    if isinstance(args.input_size, int):
        args.input_size = {d: (args.input_size, args.input_size) for d in domains}

    args.grid_sizes = {}
    for domain, size in args.input_size.items():
        args.grid_sizes[domain] = []
        for i, s in enumerate(size):
            args.grid_sizes[domain].append(s // args.patch_size[domain][i])

    args.data_path = Path(args.data_path)
    args.dataset_name = args.data_path.stem
    args.train_data_path = args.data_path / 'train'
    args.eval_data_path = args.data_path / 'val'
    if args.infer_only and args.test and args.test_data_path is None:
        args.test_data_path = args.data_path / 'test'
    if args.mapping_fn is None:
        args.mapping_fn = args.data_path / 'INFO.json'
        with open(args.mapping_fn, 'r') as f:
            original_mapping = json.load(f)
        mapping = {}
        for k, v in original_mapping.items():
            if args.ignore_index is None:
                for bg_name in ['background', 'bg']:
                    if bg_name in v['label'].lower():
                        args.ignore_index = int(k)
                        break
            mapping[v['value']] = int(k)
        args.mapping = mapping
    args.inverse_mapping = {v: k for k, v in args.mapping.items()}
    print('Mapping:')
    print(json.dumps(args.mapping, indent=2))
    if args.ignore_index is not None:
        print('-> Ignoring index', args.ignore_index)
    args.num_classes = len(args.mapping)

    args.output_dir = str(
        Path(args.base_output_dir)
        / args.version
        / args.dataset_name
    ) + '/'
    args.output_dir += Path(args.weights).stem
    if args.freeze_encoder:
        args.output_dir += '_frozen'
    args.output_dir += f'_{args.output_adapter}'
    args.output_dir += f'_{args.loss}'
    if args.minmax:
        args.output_dir += '_minmax'
    print(f">> Output dir: {args.output_dir}")

    # NOTE: Fixed out domain: segmentation
    args.out_domains = ['semseg']
    args.all_domains = list(set(args.in_domains) | set(args.out_domains))

    return args


def main(args, img_path):
    device = torch.device(args.device)

    fix_seeds(args.seed)

    model_config = None
    for kw in fm_factory.keys():
        if kw in args.weights.lower():
            model_config = fm_factory[kw]()
            break
    if model_config is None:
        raise ValueError(f"Unknown model: {args.weights}")

    # Forced minmax normalization
    if args.minmax:
        # args.norm = 'minmax'
        model_config.norm = 'minmax'

    model_config.build_domain_conf()

    if not args.show_user_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)


    # Dataset stuff
    val_transform = simple_transform(
        train=False,
        input_size=args.input_size[args.in_domains[0]][0],
        norm=model_config.norm
    )

    # Model
    input_adapters = {
        domain: model_config.domain_conf[domain]['input_adapter'](
            stride_level=model_config.domain_conf[domain]['stride_level'],
            patch_size_full=args.patch_size[domain],
            image_size=args.input_size[domain],
            learnable_pos_emb=args.learnable_pos_emb,
        )
        for domain in args.in_domains
    }

    # DPT settings are fixed for ViT-B. Modify them if using a different backbone.
    if '_base' not in model_config.model and args.output_adapter == 'dpt':
        raise NotImplementedError('Unsupported backbone: DPT head is fixed for ViT-B.')

    adapter_factory = {
        'segmenter': partial(
            SegmenterMaskTransformerAdapter,
            main_tasks=args.decoder_main_tasks.split('-'),
            embed_dim=768,
        ),
        'convnext': partial(
            ConvNeXtAdapter,
            preds_per_patch=16,
            depth=4,
            interpolate_mode=args.decoder_interpolate_mode,
            main_tasks=args.decoder_main_tasks.split('-'),
            embed_dim=6144,
        ),
        'dpt': partial(
            DPTOutputAdapter,
            stride_level=1,
            main_tasks=args.decoder_main_tasks.split('-'),
            head_type='semseg',
            embed_dim=256,
        ),
        'linear': partial(
            LinearSegAdapter,
            interpolate_mode=args.decoder_interpolate_mode,
            main_tasks=args.decoder_main_tasks.split('-')
        ),
    }

    print(f"> Using '{args.output_adapter}' output adapter")

    output_adapters = {
        'semseg': adapter_factory[args.output_adapter](
            num_classes=args.num_classes,
            patch_size=args.patch_size[args.in_domains[0]],
            task='semseg',
            image_size=args.input_size[args.in_domains[0]],
        ),
    }

    model = model_factory[model_config.model](
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        drop_path_rate=args.drop_path_encoder,
        args=args,
    )

    if args.weights:
        print('>> Loading weights from', args.weights)
        if args.weights.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.weights, map_location='cpu')
        else:
            checkpoint = torch.load(args.weights, map_location='cpu', weights_only=False)

        model = model_config(model, checkpoint)


    model.to(device)

    # print("Model =", model)

    num_layers = model.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values =", str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    optimizer = create_optimizer(
        args,
        model,
        skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None
    )
    loss_scaler = NativeScaler(enabled=args.fp16)

    if args.loss == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == 'CE-ignore-bg':
        if args.ignore_index is None:
            raise ValueError("Ignore index is not set")
        criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    elif args.loss == 'CEGDice':
        criterion = CEGDiceLoss()
    else:
        raise ValueError(f"Invalid loss: {args.loss}")
    print("criterion = ", criterion)

    # Specifies if transformer encoder should only return last layer or all layers for DPT
    return_all_layers = args.output_adapter in ['dpt']

    lookup_table = get_lookup_table(args.inverse_mapping, device=device)

    mutils.checkpoint.auto_load_model(
        args=args,
        model=model,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        best=args.test,
    )

    if args.test:
        assert img_path is not None
        test_stats = evaluate(
            model=model,
            img_path=img_path,
            val_transform=val_transform,
            device=device,
            in_domains=args.in_domains,
            fp16=args.fp16,
            return_all_layers=return_all_layers,
            log_images=True,
            infer_only=args.infer_only,
            lookup_table=lookup_table
        )
        return test_stats


def get_lookup_table(mapping: Dict[int, int], device: torch.device) -> torch.Tensor:
    # Create a lookup table
    max_key = max(mapping.keys())  # Get the largest key in the dictionary
    lookup_table = torch.full((max_key + 1,), -1)  # Initialize with default values
    for key, value in mapping.items():
        lookup_table[key] = value  # Fill the lookup table
    lookup_table = lookup_table.to(device)
    return lookup_table


@torch.no_grad()
def evaluate(
    model,
    img_path,
    val_transform,
    device,
    in_domains,
    fp16=True,
    return_all_layers=False,
    log_images=False,
    infer_only=False,
    lookup_table=None
):
    # Switch to evaluation mode
    model.eval()

    save_dir = Path("__image_seg")

    img_pil = Image.open(img_path).convert("L")
    img_np = np.array(img_pil)
    out = val_transform(image=img_np)
    img_tensor = out["image"]          # (C, H, W)
    img_tensor = img_tensor.unsqueeze(0)  # (1, C, H, W)
    img_tensor = img_tensor.to(device, non_blocking=True)
    input_dict = {in_domains[0]: img_tensor}

    # Forward + backward
    with autocast('cuda', enabled=fp16):
        preds = model(input_dict, return_all_layers=return_all_layers)
        seg_pred = preds["semseg"]

    # If there is void, exclude it from the preds and take second highest class
    seg_pred_argmax = seg_pred.argmax(dim=1)
    if infer_only or log_images:
        assert save_dir is not None
        assert lookup_table is not None
        pred_i = lookup_table[seg_pred_argmax.long()].cpu().numpy().astype(np.uint8)
        path = Path(img_path)
        filename = f'{path.stem}_pred.png'
        io.imsave(save_dir / filename, pred_i)
        save_path = save_dir / filename

    if infer_only:
        print('Inference done. Exiting...')

    return save_path


def seg_oct(image_url):
    opts = get_args()
    if opts.output_dir:
        if opts.minmax:
            opts.output_dir += '_minmax'
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    img_path = download(image_url)
    save_path = main(opts, img_path)
    return save_path

if __name__ == '__main__':
    img_path = "__image/NORMAL_7_9.png"
    opts = get_args()
    if opts.output_dir:
        if opts.minmax:
            opts.output_dir += '_minmax'
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    save_path = main(opts, img_path)
    print(save_path)
