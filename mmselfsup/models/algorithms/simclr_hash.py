# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import numpy as np
import torch

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from ..utils import GatherLayer
from .base import BaseModel


@MODELS.register_module()
class SimCLR_Hash(BaseModel):
    """SimCLR.

    Implementation of `A Simple Framework for Contrastive Learning of Visual
    Representations <https://arxiv.org/abs/2002.05709>`_.
    """

    @staticmethod
    def _create_buffer(
        batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the mask and the index of positive samples.

        Args:
            batch_size (int): The batch size.
            device (torch.device): The device of backend.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - The mask for feature selection.
            - The index of positive samples.
            - The mask of negative samples.
        """
        mask = 1 - torch.eye(batch_size * 2, dtype=torch.uint8).to(device)
        pos_idx = (
            torch.arange(batch_size * 2).to(device),
            2 * torch.arange(batch_size, dtype=torch.long).unsqueeze(1).repeat(
                1, 2).view(-1, 1).squeeze().to(device))
        neg_mask = torch.ones((batch_size * 2, batch_size * 2 - 1),
                              dtype=torch.uint8).to(device)
        neg_mask[pos_idx] = 0
        return mask, pos_idx, neg_mask

    def extract_feat(self, inputs: List[torch.Tensor],
                     **kwargs) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            inputs (List[torch.Tensor]): The input images.

        Returns:
            Tuple[torch.Tensor]: Backbone outputs.
        """
        x = self.backbone(inputs[0])

        return x

    def loss(self, inputs: List[torch.Tensor,],

            data_samples: List[SelfSupDataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        assert isinstance(inputs, list)
        inputs = torch.stack(inputs, 1)
        inputs = inputs.reshape((inputs.size(0) * 2, inputs.size(2),
                                 inputs.size(3), inputs.size(4)))
        # mask = torch.stack(
        #     [data_sample.mask.value for data_sample in data_samples])
        #
        # mask1 = torch.stack(
        #     [data_sample.mask1.value for data_sample in data_samples])
        #
        # mask=torch.cat([mask,mask1],dim=0)

        x = self.backbone(inputs)
        z = self.neck(x)[0]  # (2n)xd
        gamma=np.sqrt(self.epoch+1)

        z=torch.tanh(gamma*z)
        b=z

        z = z / (torch.norm(z, p=2, dim=1, keepdim=True) + 1e-10)
        z = torch.cat(GatherLayer.apply(z), dim=0)  # (2N)xd
        assert z.size(0) % 2 == 0
        N = z.size(0) // 2
        s = torch.matmul(z, z.permute(1, 0))  # (2N)x(2N)
        mask2, pos_idx, neg_mask = self._create_buffer(N, s.device)

        # remove diagonal, (2N)x(2N-1)
        s = torch.masked_select(s, mask2 == 1).reshape(s.size(0), -1)
        positive = s[pos_idx].unsqueeze(1)  # (2N)x1

        # select negative, (2N)x(2N-2)
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)

        loss_1 = self.head(b,positive, negative)

        losses = dict(loss=loss_1)
        return losses

    def predict(self, inputs: List[torch.Tensor],

                data_samples: List[SelfSupDataSample],
                 **kwargs) -> Dict[str, torch.Tensor]:
            """The forward function in training.

            Args:
                inputs (List[torch.Tensor]): The input images.
                data_samples (List[SelfSupDataSample]): All elements required
                    during the forward function.

            Returns:
                Dict[str, torch.Tensor]: A dictionary of loss components.
            """
            x = self.backbone(inputs[0])
            z = self.neck(x)[0]  # (2n)xd

            hash=torch.sign(z)

            return hash

    def set_epoch(self,epoch):
        self.epoch=epoch