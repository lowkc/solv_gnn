import warnings
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss that weighs each element differently.
    Weight will be multiplied to the squares of difference.
    All settings are the same as the :meth:`torch.nn.MSELoss`, except that if
    `reduction` is `mean`, the loss will be divided by the sum of `weight`, instead of
    the batch size.
    The weight could be used to ignore some elements in `target` by setting the
    corresponding elements in weight to 0, and it can also be used to scale the element
    differently.
    Args:
        weight (Tensor): is weight is `None` this behaves exactly the same as
        :meth:`torch.nn.MSELoss`.
    """

    def __init__(self, reduction="mean"):
        self.reduction = reduction
        super(WeightedMSELoss, self).__init__()

    def forward(self, input, target, weight):

        if weight is None:
            return F.l1_loss(input, target, reduction=self.reduction)
        else:
            if input.size() != target.size() != weight.size():
                warnings.warn(
                    "Input size ({}) is different from the target size ({}) or weight "
                    "size ({}). This will likely lead to incorrect results due "
                    "to broadcasting. Please ensure they have the same size.".format(
                        input.size(), target.size(), weight.size()
                    )
                )

            rst = ((input - target) ** 2) * weight
            if self.reduction != "none":
                if self.reduction == "mean":
                    rst = torch.sum(rst) / torch.sum(weight)
                else:
                    rst = torch.sum(rst)

            return rst


class WeightedL1Loss(nn.Module):
    """
    Weighted L1 Loss that weighs each element differently.
    Weight will be multiplied to the squares of difference.
    All settings are the same as the :meth:`torch.nn.L1Loss`, except that if
    `reduction` is `mean`, the loss will be divided by the sum of `weight`, instead of
    the batch size.
    The weight could be used to ignore some elements in `target` by setting the
    corresponding elements in weight to 0, and it can also be used to scale the element
    differently.
    Args:
        weight (Tensor): is weight is `None` this behaves exactly the same as
        :meth:`torch.nn.L1Loss`.
    """

    def __init__(self, reduction="mean"):
        self.reduction = reduction
        super(WeightedL1Loss, self).__init__()

    def forward(self, input, target, weight):

        if weight is None:
            return F.l1_loss(input, target, reduction=self.reduction)
        else:
            if input.size() != target.size() != weight.size():
                warnings.warn(
                    "Input size ({}) is different from the target size ({}) or weight "
                    "size ({}). This will likely lead to incorrect results due "
                    "to broadcasting. Please ensure they have the same size.".format(
                        input.size(), target.size(), weight.size()
                    )
                )

            rst = torch.abs(input - target) * weight
            if self.reduction != "none":
                if self.reduction == "mean":
                    rst = torch.sum(rst) / torch.sum(weight)
                else:
                    rst = torch.sum(rst)

            return rst


class EarlyStopping:
    def __init__(self, patience=200, silent=True):
        self.patience = patience
        self.silent = silent
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if not self.silent:
                print("EarlyStopping counter: {}/{}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"