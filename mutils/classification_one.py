from collections import OrderedDict
import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from PIL import Image
from torchvision import transforms


@torch.no_grad()
def evaluate_one(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    ):

    # switch to evaluation mode
    model.eval()

    # 讀取圖片（PIL Image）
    img = Image.open(image_path).convert("L")

    # 定義轉換流程
    transform = transforms.Compose([
        transforms.Resize((512, 512)),   # 視模型需求
        transforms.ToTensor(),            # 轉成 torch.Tensor
    ])

    # 轉成 Tensor
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    image = img_tensor.to(device, non_blocking=True)

    with autocast('cuda', enabled=True):
        print(image.shape)
        output = model(image)

        prediction_softmax = nn.Softmax(dim=1)(output)
        _, prediction_decode = torch.max(prediction_softmax, 1)
        # prediction_decode = 1

    return OrderedDict({
        'Prediction': prediction_decode,
        'Confidence': prediction_softmax
    })

