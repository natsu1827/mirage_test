import albumentations as A
from albumentations.pytorch import ToTensorV2

def simple_transform(
    train: bool,
    # additional_targets: Dict[str, str],
    input_size: int = 512,
    norm: str = 'minmax',
):
    """Default transform for semantic segmentation, applied on all
    modalities.
    """

    norm_list = []
    if norm == 'imagenet':
        print("Using imagenet normalization")
        norm_list += [
            ToRGB(p=1),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    elif norm == 'sam':
        print("Using SAM normalization")
        # Rescale everything to [0, 255]
        norm_list += [
            ToRGB(p=1),
            ToRange(range=(0, 255), p=1),
        ]
    elif norm == 'z-score':
        print("Using z-score normalization")
        norm_list += [
            ToRGB(p=1),
            A.Normalize(mean=0, std=1),
        ]
    else:
        # No extra operations needed
        pass

    transform_list = [
        A.Resize(height=input_size, width=input_size, p=1),
    ]
    transform_list += norm_list
    transform_list += [
        A.ToFloat(max_value=255.0),  # ✅ 確保輸出 float
        ToTensorV2(),
    ]
    transform = A.Compose(transform_list)  # type: ignore

    print(
        f'[Train: {train}]'
        f' [Input size: {input_size}]'
        f' [Normalization: {norm}]'
        f' [Transform list: {transform_list}]'
    )
    return transform