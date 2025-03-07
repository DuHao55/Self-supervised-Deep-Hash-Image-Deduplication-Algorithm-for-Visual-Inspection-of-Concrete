# Copyright (c) OpenMMLab. All rights reserved.
from .beit_vit import BEiTViT
from .cae_vit import CAEViT
from .mae_vit import MAEViT
from .maskfeat_vit import MaskFeatViT
from .milan_vit import MILANViT
from .mixmim_backbone import MixMIMTransformerPretrain
from .mocov3_vit import MoCoV3ViT
from .resnet import ResNet, ResNetSobel, ResNetV1d
from .resnext import ResNeXt
from .simmim_swin import SimMIMSwinTransformer
from .simmim_swin_FPN import SimMIMSwinTransformer_FPN
from .simclr_swin_FPN import SimCLRSwinTransformer_FPN
from .simclr_swin import SimCLRSwinTransformer
__all__ = [
    'SimMIMSwinTransformer_FPN', 'SimCLRSwinTransformer_FPN',
]
