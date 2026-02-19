"""
Pretrained convolutional image encoder based on MobileNetV2.
The backbone is frozen by default; only the projection layer is trained.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models


class ImageEncoder(nn.Module):
    """
    MobileNetV2-based image encoder.

    Input:  [B, 3, 224, 224] ImageNet-normalized tensor
    Output: [B, output_dim] embedding vector

    Output size is independent of input image resolution:
    AdaptiveAvgPool2d collapses any spatial size to 1x1.

    Args:
        output_dim: dimensionality of the output embedding
        freeze:     if True, freeze backbone weights (train only projection layer)
    """

    def __init__(self, output_dim: int, freeze: bool = True):
        super().__init__()

        mobilenet = tv_models.mobilenet_v2(
            weights=tv_models.MobileNet_V2_Weights.DEFAULT
        )
        self.backbone = mobilenet.features        # [B, 1280, H', W']
        self.pool     = nn.AdaptiveAvgPool2d(1)  # [B, 1280, 1, 1]
        self.proj     = nn.Linear(1280, output_dim)  # [B, output_dim]

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"[INFO] ImageEncoder: backbone frozen. "
                  f"Trainable: projection layer (1280 -> {output_dim})")
        else:
            print(f"[INFO] ImageEncoder: full fine-tuning enabled. "
                  f"Output dim: {output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)    # [B, 1280, H', W']
        pooled   = self.pool(features) # [B, 1280, 1, 1]
        flat     = pooled.flatten(1)   # [B, 1280]
        return self.proj(flat)         # [B, output_dim]


def preprocess_image(
    img_bgr: np.ndarray,
    device: torch.device,
    mean: list,
    std: list,
) -> torch.Tensor:
    """
    Convert a BGR numpy image to an ImageNet-normalized tensor for MobileNetV2.

    Args:
        img_bgr: numpy array [H, W, 3] uint8, BGR color space
        device:  target device (cpu / cuda)
        mean:    ImageNet channel means [R, G, B]
        std:     ImageNet channel standard deviations [R, G, B]

    Returns:
        torch.Tensor of shape [1, 3, 224, 224]
    """
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr  = np.array(std,  dtype=np.float32)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img     = cv2.resize(img_rgb, (224, 224)).astype(np.float32) / 255.0
    img     = (img - mean_arr) / std_arr
    tensor  = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)
