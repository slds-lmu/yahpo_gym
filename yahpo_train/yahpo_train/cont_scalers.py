import torch
import torch.nn as nn
from functools import partial


class ContTransformerNone(nn.Module):
    """
    Transformer for Continuous Variables. Performs no transformation.
    """

    def __init__(self, x, **kwargs):
        super().__init__()

    @staticmethod
    def forward(x, **kwargs):
        """
        Batch-wise transform for x.
        """
        return x.float()

    @staticmethod
    def invert(x, **kwargs):
        """
        Batch-wise inverse transform for x.
        """
        return x.float()


class ContTransformerRange(nn.Module):
    """
    Transformer for Continuous Variables. Transforms to [0,1].
    """

    def __init__(self, x, **kwargs):
        super().__init__()
        self.min, self.max = torch.min(x[~torch.isnan(x)]), torch.max(
            x[~torch.isnan(x)]
        )
        if self.max == self.min:
            raise Exception("Constant feature detected!")

    def forward(self, x, **kwargs):
        """
        Batch-wise transform for x.
        """
        x = (x - self.min) / (self.max - self.min)
        return x.float()

    def invert(self, x, **kwargs):
        """
        Batch-wise inverse transform for x.
        """
        x = x * (self.max - self.min) + self.min
        return x.float()


class ContTransformerStandardize(nn.Module):
    """
    Transformer for Continuous Variables. Transforms via standardization.
    """

    def __init__(self, x, robust=False, **kwargs):
        super().__init__()
        if robust:
            self.center, self.scale = torch.median(x[~torch.isnan(x)]), torch.quantile(
                x[~torch.isnan(x)], 0.75
            ) - torch.quantile(x[~torch.isnan(x)], 0.25)
        else:
            self.center, self.scale = torch.mean(x[~torch.isnan(x)]), torch.sqrt(
                torch.var(x[~torch.isnan(x)])
            )
        if self.scale <= 1e-12:
            raise Exception("Constant feature detected!")

    def forward(self, x, **kwargs):
        """
        Batch-wise transform for x.
        """
        x = (x - self.center) / self.scale
        return x.float()

    def invert(self, x, **kwargs):
        """
        Batch-wise inverse transform for x.
        """
        x = x * self.scale + self.center
        return x.float()


class ContTransformerStandardizeGrouped(nn.Module):
    """
    Transformer for Continuous Variables. Transforms via standardization.
    Grouped by group.
    """

    def __init__(self, x, group, robust=False, **kwargs):
        super().__init__()
        self.group_ids = torch.unique(group)

        if robust:
            self.centers = torch.stack(
                [
                    torch.median(
                        x[group == group_id][~torch.isnan(x[group == group_id])]
                    )
                    for group_id in self.group_ids
                ]
            )
            self.scales = torch.stack(
                [
                    torch.quantile(
                        x[group == group_id][~torch.isnan(x[group == group_id])], 0.75
                    )
                    - torch.quantile(
                        x[group == group_id][~torch.isnan(x[group == group_id])], 0.25
                    )
                    for group_id in self.group_ids
                ]
            )
        else:
            self.centers = torch.stack(
                [
                    torch.mean(x[group == group_id][~torch.isnan(x[group == group_id])])
                    for group_id in self.group_ids
                ]
            )
            self.scales = torch.stack(
                [
                    torch.sqrt(
                        torch.var(
                            x[group == group_id][~torch.isnan(x[group == group_id])]
                        )
                    )
                    for group_id in self.group_ids
                ]
            )

        if any([scale <= 1e-12 for scale in self.scales]):
            raise Exception("Constant feature detected!")

    def forward(self, x, group, **kwargs):
        """
        Batch-wise transform for x.
        """
        centers = (
            self.centers.to(x.device).index_select(dim=0, index=group - 1).to(x.device)
        )
        scales = (
            self.scales.to(x.device).index_select(dim=0, index=group - 1).to(x.device)
        )
        x = (x - centers) / scales
        return x.float()

    def invert(self, x, group, **kwargs):
        """
        Batch-wise inverse transform for x.
        """
        centers = (
            self.centers.to(x.device).index_select(dim=0, index=group - 1).to(x.device)
        )
        scales = (
            self.scales.to(x.device).index_select(dim=0, index=group - 1).to(x.device)
        )
        x = x * scales + centers
        return x.float()


class ContTransformerInt(nn.Module):
    """

    Transform doubles to their nearest integer.
    Assumes that the input is integer. Therefore, the forward step is simply the identity.
    Note that the data type remains unchanged.

    """

    def __init__(self, x, **kwargs):
        super().__init__()

    @staticmethod
    def forward(x, **kwargs):
        """
        Identity.
        """
        return x.float()

    @staticmethod
    def invert(x, **kwargs):
        """
        Round to nearest integer.
        """
        x = torch.round(x)
        return x.float()


class ContTransformerClamp(nn.Module):
    """
    Transformer for Continuous Variables. Transforms to [min,max].
    """

    def __init__(self, x, min=None, max=None, **kwargs):
        super().__init__()
        self.min, self.max = min, max
        if self.min is not None:
            self.min = torch.Tensor([min])
        if self.max is not None:
            self.max = torch.Tensor([max])

    @staticmethod
    def forward(x, **kwargs):
        """
        Batch-wise transform for x.
        """
        return x.float()

    def invert(self, x, **kwargs):
        """
        Batch-wise inverse transform for x.
        """
        if self.min is not None:
            min = self.min.to(x.device)
        else:
            min = None
        if self.max is not None:
            max = self.max.to(x.device)
        else:
            max = None
        x = torch.clamp(x, min, max)
        return x.float()


class ContTransformerClampGrouped(nn.Module):
    """
    Transformer for Continuous Variables. Transforms to [min,max].
    Grouped by group.
    """

    def __init__(self, x, group, min=None, max=None, **kwargs):
        super().__init__()
        self.group_ids = torch.unique(
            group
        )  # see also the resulting encoding.json in the data_path directories how groups will match to ids
        self.min, self.max = min, max
        if self.min is not None:
            if len(min) != len(self.group_ids):
                raise Exception("Length of min does not match number of groups.")
            self.min = torch.Tensor([min]).squeeze(0)
        if self.max is not None:
            if len(max) != len(self.group_ids):
                raise Exception("Length of max does not match number of groups.")
            self.max = torch.Tensor([max]).squeeze(0)

    @staticmethod
    def forward(x, **kwargs):
        """
        Batch-wise transform for x.
        """
        return x.float()

    def invert(self, x, group, **kwargs):
        """
        Batch-wise inverse transform for x.
        """
        if self.min is not None:
            min = (
                self.min.to(x.device).index_select(dim=0, index=group - 1).to(x.device)
            )
        else:
            min = None
        if self.max is not None:
            max = (
                self.max.to(x.device).index_select(dim=0, index=group - 1).to(x.device)
            )
        else:
            max = None
        x = torch.clamp(x, min, max)
        return x.float()


class ContTransformerChain(nn.Module):
    """
    Chained transformer Continuous Variables. Chains several transforms.
    During forward pass, transforms are applied according to the list order,
    during invert, the order is reversed.
    """

    def __init__(self, x, tfms, **kwargs):
        super().__init__()
        self.tfms = []
        for tf in tfms:
            itf = tf(x, **kwargs)
            x = itf.forward(x, **kwargs)  # test whether forward works
            self.tfms += [itf]

    def forward(self, x, **kwargs):
        """
        Chained batch-wise transform for x.
        """
        for tf in self.tfms:
            x = tf.forward(x, **kwargs)
        return x

    def invert(self, x, **kwargs):
        """
        Chained batch-wise inverse transform for x.
        """
        for tf in reversed(self.tfms):
            x = tf.invert(x, **kwargs)
        return x.float()


def tfms_chain(tfms):
    return partial(ContTransformerChain, tfms=tfms)


ContTransformerStandardizeRange = tfms_chain(
    [ContTransformerStandardize, ContTransformerRange]
)

ContTransformerStandardizeGroupedRange = tfms_chain(
    [ContTransformerStandardizeGrouped, ContTransformerRange]
)
