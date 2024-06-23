import torch
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from torch import nn
from typing import Tuple

class DC_BCE_and_DA_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, da_kwargs, weight_ce=1, weight_dice=1, weight_da=1e-2, use_ignore_label: bool = False,
                 masked_label: int = None,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_BCE_and_DA_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_da = weight_da
        self.use_ignore_label = use_ignore_label
        self.masked_label = masked_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)
        self.da = RobustCrossEntropyLoss(**da_kwargs)

    def forward(self, global_net_output: Tuple[torch.Tensor], global_target: Tuple[torch.Tensor]):
        """

        Parameters
        ----------
        global_net_output: Tuple[torch.Tensor]
            Tuple of segmentation net_output ( [B,N,:]) and domain_pred ([B,])
        global_target: Tuple[torch.Tensor]
            Tuple of segmentation target ([B,N,:]) and domain_target ([B,])
        Returns
        -------

        """
        net_output, domain_pred = global_net_output
        target, domain_target = global_target
        if self.use_ignore_label:
            self.masked_label = 0
            mask = torch.ones(target.shape, dtype=torch.bool)
            mask[domain_target == self.masked_label, :] = 0
            target_regions = target
        else:
            target_regions = target
            mask = None

        
        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)

        # JR: implement DA (domain adaptation) loss: one-hot domain_pred and domain_target
        da_loss = self.da(domain_pred, domain_target)

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_da * da_loss
        return result


class DC_BCE_masked_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 masked_label: int = None,
                 dice_class=MemoryEfficientSoftDiceLoss):

        super(DC_BCE_masked_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label
        self.masked_label = masked_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output, target, domain_target):
        if self.use_ignore_label:
            self.masked_label = 0
            mask_size = list(target.shape)
            mask_size[1] = 1
            mask = torch.ones(mask_size, dtype=torch.bool).to("cuda")
            mask[domain_target == self.masked_label, :] = 0
            target_regions = target
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
