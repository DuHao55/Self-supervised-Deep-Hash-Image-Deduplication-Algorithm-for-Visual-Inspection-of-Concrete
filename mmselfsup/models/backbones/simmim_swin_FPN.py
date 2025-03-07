# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmcls.models import SwinTransformer
from mmengine.model.weight_init import trunc_normal_
import torch.nn.functional as F
from mmselfsup.registry import MODELS
from typing import List
class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self,  in_channels=[192,192,192,192],embedding_dim=192,dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        # self.linear_fuse = ConvModule(
        #     c1=embedding_dim * 4,
        #     c2=embedding_dim,
        #     k=1,
        # )

        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c1 = F.interpolate(_c1, size=c4.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c4.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c4.size()[2:], mode='bilinear', align_corners=False)

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])

        _c = torch.cat([_c4, _c3, _c2, _c1], dim=1)

        x = self.dropout(_c)


        return x


class ConvModule(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True, inplace=False):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.inplace = inplace

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))



class FPN(nn.Module):
    """
    Feature Pyramid Network.

    This is an implementation of - Feature Pyramid Networks for Object
    Detection (https://arxiv.org/abs/1612.03144)

    Args:
        in_channels (List[int]):
            number of input channels per scale

        out_channels (int):
            number of output channels (used at each scale)

        num_outs (int):
            number of output scales

        start_level (int):
            index of the first input scale to use as an output scale

        end_level (int, default=-1):
            index of the last input scale to use as an output scale

    Example:

        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=4,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 ):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                k=1,
                act=True,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                p=1,
                act=True,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    s=2,
                    p=1,
                    act=True,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)


    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

@MODELS.register_module()
class SimMIMSwinTransformer_FPN(SwinTransformer):
    """Swin Transformer for SimMIM.

    Args:
        Args:
        arch (str | dict): Swin Transformer architecture
            Defaults to 'T'.
        img_size (int | tuple): The size of input image.
            Defaults to 224.
        in_channels (int): The num of input channels.
            Defaults to 3.
        drop_rate (float): Dropout rate after embedding.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate.
            Defaults to 0.1.
        out_indices (tuple): Layers to be outputted. Defaults to (3, ).
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults to False.
        with_cp (bool): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Defaults to False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        norm_cfg (dict): Config dict for normalization layer at end
            of backone. Defaults to dict(type='LN')
        stage_cfgs (Sequence | dict): Extra config dict for each
            stage. Defaults to empty dict.
        patch_cfg (dict): Extra config dict for patch embedding.
            Defaults to empty dict.
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 arch: Union[str, dict] = 'T',
                 img_size: Union[Tuple[int, int], int] = 224,
                 in_channels: int = 3,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.1,
                 #out_indices: tuple = (3,),
                 out_indices: tuple = (0,1,2,3),  #change this control outputs当你创建只有一个元素的元组时，需要在元素后面加上逗号，这是为了区分括号的意义
                 use_abs_pos_embed: bool = False,
                 with_cp: bool = False,
                 frozen_stages: bool = -1,
                 norm_eval: bool = False,
                 norm_cfg: dict = dict(type='LN'),
                 stage_cfgs: Union[Sequence, dict] = dict(),
                 patch_cfg: dict = dict(),
                 pad_small_map: bool = False,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(
            arch=arch,
            img_size=img_size,
            in_channels=in_channels,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            out_indices=out_indices,
            use_abs_pos_embed=use_abs_pos_embed,
            with_cp=with_cp,
            frozen_stages=frozen_stages,
            norm_eval=norm_eval,
            norm_cfg=norm_cfg,
            stage_cfgs=stage_cfgs,
            patch_cfg=patch_cfg,
            pad_small_map=pad_small_map,
            init_cfg=init_cfg)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        self.fusion_feature=FPN([96,96*2,96*2**2,96*2**3],96*2,4)
        self.SegFormerHead=SegFormerHead(dropout_ratio=0.1)
    def init_weights(self) -> None:
        """Initialize weights."""
        super().init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            return

        if self.use_abs_pos_embed:
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        trunc_normal_(self.mask_token, mean=0, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor) -> Sequence[torch.Tensor]:
        """Generate features for masked images.

        This function generates mask images and get the hidden features for
        them.

        Args:
            x (torch.Tensor): Input images.
            mask (torch.Tensor): Masks used to construct masked images.

        Returns:
            tuple: A tuple containing features from multi-stages.
        """
        x, hw_shape = self.patch_embed(x)    #input(b, c, h, w)->(b, h*w,c)

        assert mask is not None   #mask (8,64,64)
        B, L, _ = x.shape

        mask_token = self.mask_token.expand(B, L, -1)  #-1表示当前维度不变，mask_token reshape （B,L,embed_dims),same as x(patch_embed)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)  #mask=(b,h,w)->(b,h*w)->(b,h*w,1)  w=(8,4096,1),4096=64*64
        x = x * (1. - w) + mask_token * w  # x=(8,4096,96)  w type_as mask_token,also can trainable,
        # w=0,x=x(where no mask,no need train and grad);
        # w=1,x=mask_token(where has mask, need train to access to raw pixel,w just is 0 or 1,no need change,mask_token is trained);

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed

        x = self.drop_after_pos(x)

        outs = []
        outs2=[]
        for i, stage in enumerate(self.stages):
            x, hw_shape = stage(
                x, hw_shape, do_downsample=self.out_after_downsample)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                out = out.view(-1, *hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)
            if stage.downsample is not None and not self.out_after_downsample:
                x, hw_shape = stage.downsample(x, hw_shape)

        outs1=self.fusion_feature.forward(outs)
        outs1=self.SegFormerHead.forward(outs1)
        outs2.append(outs1)
        return tuple(outs2)







