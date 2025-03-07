# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class SimMIMNeck(BaseModule):
    """Pre-train Neck For SimMIM.

    This neck reconstructs the original image from the shrunk feature map.

    Args:
        in_channels (int): Channel dimension of the feature map.
        encoder_stride (int): The total stride of the encoder.
    """

    def __init__(self, in_channels: int, encoder_stride: int) -> None:
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=encoder_stride**2 * 3,
                kernel_size=1),
            nn.PixelShuffle(encoder_stride),
            #(8,768,16,16) conv-> (8,3072,16,16)->PixelShuffle(8,3,16*encoder_stride,16*encoder_stride)=(8,3,512,512)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        x = self.decoder(x)

        return x
