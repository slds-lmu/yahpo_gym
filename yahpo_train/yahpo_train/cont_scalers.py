from functools import partial
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import boxcox


class ContTransformerNone(nn.Module):
    """
    Transformer for Continuous Variables. Performs no transformation.
    """

    def __init__(self, x: torch.Tensor, **kwargs):
        super().__init__()

    @staticmethod
    def forward(x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Batch-wise transform for x.
        """
        return x.float()

    @staticmethod
    def invert(x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Batch-wise inverse transform for x.
        """
        return x.float()


class ContTransformerRange(nn.Module):
    """
    Transformer for Continuous Variables. Transforms to [eps,1-eps].
    If x_range is "-1-1", transforms to [-1+eps,1-eps].
    """

    def __init__(
        self,
        x: torch.Tensor,
        x_id: str,
        eps: float = 1e-2,
        x_range: str = "0-1",
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.x_range = x_range
        self.min, self.max = torch.min(x[~torch.isnan(x)]), torch.max(
            x[~torch.isnan(x)]
        )
        if self.max == self.min:
            raise Exception(f"Constant variable `{x_id}` detected!")

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Batch-wise transform for x.
        """
        if self.x_range == "0-1":
            x = self.eps + (1 - 2 * self.eps) * (x - self.min) / (self.max - self.min)
        elif self.x_range == "-1-1":
            x = self.eps + (1 - 2 * self.eps) * (x - self.min) / (self.max - self.min)
            x = 2 * x - 1
        return x.float()

    def invert(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Batch-wise inverse transform for x.
        """
        if self.x_range == "0-1":
            x = self.min + (x - self.eps) / (1 - 2 * self.eps) * (self.max - self.min)
        elif self.x_range == "-1-1":
            x = (x + 1) / 2
            x = self.min + (x - self.eps) / (1 - 2 * self.eps) * (self.max - self.min)
        return x.float()


class ContTransformerRangeGrouped(nn.Module):
    """
    Transformer for Continuous Variables. Transforms to [eps,1-eps].
    If x_range is "-1-1", transforms to [-1+eps,1-eps].
    Grouped by group.
    """

    def __init__(
        self,
        x: torch.Tensor,
        group: torch.Tensor,
        x_id: str,
        eps: float = 1e-2,
        x_range: str = "0-1",
        **kwargs,
    ):
        super().__init__()
        self.group_ids = torch.unique(group)
        self.eps = eps
        self.x_range = x_range

        self.mins = torch.stack(
            [
                torch.min(x[group == group_id][~torch.isnan(x[group == group_id])])
                for group_id in self.group_ids
            ]
        )

        self.maxs = torch.stack(
            [
                torch.max(x[group == group_id][~torch.isnan(x[group == group_id])])
                for group_id in self.group_ids
            ]
        )

        if any([max_ == min_ for max_, min_ in zip(self.maxs, self.mins)]):
            raise Exception(f"Constant variable `{x_id}` detected!")

    def forward(self, x: torch.Tensor, group: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Batch-wise transform for x.
        """
        mins = self.mins.to(x.device).index_select(dim=0, index=group - 1).to(x.device)
        maxs = self.maxs.to(x.device).index_select(dim=0, index=group - 1).to(x.device)

        if self.x_range == "0-1":
            x = self.eps + (1 - 2 * self.eps) * (x - mins) / (maxs - mins)
        elif self.x_range == "-1-1":
            x = self.eps + (1 - 2 * self.eps) * (x - mins) / (maxs - mins)
            x = 2 * x - 1
        return x.float()

    def invert(self, x: torch.Tensor, group: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Batch-wise inverse transform for x.
        """
        mins = self.mins.to(x.device).index_select(dim=0, index=group - 1).to(x.device)
        maxs = self.maxs.to(x.device).index_select(dim=0, index=group - 1).to(x.device)

        if self.x_range == "0-1":
            x = mins + (x - self.eps) / (1 - 2 * self.eps) * (maxs - mins)
        elif self.x_range == "-1-1":
            x = (x + 1) / 2
            x = mins + (x - self.eps) / (1 - 2 * self.eps) * (maxs - mins)
        return x.float()


class ContTransformerStandardize(nn.Module):
    """
    Transformer for Continuous Variables. Transforms via standardization.
    """

    def __init__(self, x: torch.Tensor, x_id: str, robust: bool = False, **kwargs):
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
            raise Exception(f"Constant variable `{x_id}` detected!")

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Batch-wise transform for x.
        """
        x = (x - self.center) / self.scale
        return x.float()

    def invert(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
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

    def __init__(
        self,
        x: torch.Tensor,
        group: torch.Tensor,
        x_id: str,
        robust: bool = False,
        **kwargs,
    ):
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
            raise Exception(f"Constant variable `{x_id}` detected!")

    def forward(self, x: torch.Tensor, group: torch.Tensor, **kwargs) -> torch.Tensor:
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

    def invert(self, x: torch.Tensor, group: torch.Tensor, **kwargs) -> torch.Tensor:
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


class ContTransformerBoxCox(nn.Module):
    """
    Transformer for Continuous Variables. Transforms via Box-Cox transformation.
    """

    def __init__(self, x: torch.Tensor, x_id: str, **kwargs):
        super().__init__()
        self.fallback = False

        # convert PyTorch tensor to numpy array for scipy
        x_np = x.detach().cpu().numpy()

        # estimate the Box-Cox transformation parameter
        with np.errstate(all="raise"):
            try:
                _, self.lmbda_ = boxcox(x_np[~np.isnan(x_np)])
            except Exception as e:
                print(
                    f"Box-Cox estimation for variable `{x_id}` failed with error: {e}. Using identity as fallback."
                )
                self.fallback = True

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Batch-wise transform for x.
        """
        if self.fallback:
            return x
        else:
            if self.lmbda_ != 0:
                x = (torch.pow(x, self.lmbda_) - 1) / self.lmbda_
            else:
                x = torch.log(x)
            return x.float()

    def invert(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Batch-wise inverse transform for x.
        """
        if self.fallback:
            return x
        else:
            if self.lmbda_ != 0:
                x = torch.pow((self.lmbda_ * x) + 1, 1 / self.lmbda_)
            else:
                x = torch.exp(x)
            return x.float()


class ContTransformerInt(nn.Module):
    """
    Transform doubles to their nearest integer.
    Assumes that the input is integer. Therefore, the forward step is simply the identity.
    Note that the data type remains unchanged as a float.
    """

    def __init__(self, x: torch.Tensor, **kwargs):
        super().__init__()

    @staticmethod
    def forward(x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Identity.
        """
        return x.float()

    @staticmethod
    def invert(x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Round to nearest integer.
        """
        x = torch.round(x)
        return x.float()


class ContTransformerClamp(nn.Module):
    """
    Transformer for Continuous Variables. Transforms to [min,max].
    """

    def __init__(
        self,
        x: torch.Tensor,
        min: Optional[float] = None,
        max: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        self.min, self.max = min, max
        if self.min is not None:
            self.min = torch.Tensor([min])
        if self.max is not None:
            self.max = torch.Tensor([max])

    @staticmethod
    def forward(x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Batch-wise transform for x.
        """
        return x.float()

    def invert(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
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

    def __init__(
        self,
        x: torch.Tensor,
        group: torch.Tensor,
        min: Optional[List[float]] = None,
        max: Optional[List[float]] = None,
        **kwargs,
    ):
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
    def forward(x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Batch-wise transform for x.
        """
        return x.float()

    def invert(self, x: torch.Tensor, group: torch.Tensor, **kwargs) -> torch.Tensor:
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


class ContTransformerLog(nn.Module):
    """
    Transformer for Continuous Variables. Transforms to log-scale.
    """

    def __init__(self, x: torch.Tensor, x_id: str, **kwargs):
        super().__init__()
        if torch.min(x) <= 0:
            raise Exception(
                f"Log transformation requires strictly positive values. Failed for variable `{x_id}`."
            )

    @staticmethod
    def forward(x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Batch-wise transform for x.
        """
        return torch.log(x.float())

    @staticmethod
    def invert(x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Batch-wise inverse transform for x.
        """
        return torch.exp(x.float())


class ContTransformerShift(nn.Module):
    """
    Transformer for Continuous Variables. Shifts by a constant.
    """

    def __init__(self, x: torch.Tensor, shift: float = 1e-8, **kwargs):
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Batch-wise transform for x.
        """
        return x.float() + self.shift

    def invert(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Batch-wise inverse transform for x.
        """
        return x.float() - self.shift


class ContTransformerChain(nn.Module):
    """
    Chained transformer Continuous Variables. Chains several transforms.
    During forward pass, transforms are applied according to the list order,
    during invert, the order is reversed.
    """

    def __init__(self, x: torch.Tensor, tfms: List[nn.Module], **kwargs):
        super().__init__()
        self.tfms = []
        for tf in tfms:
            itf = tf(x, **kwargs)
            x = itf.forward(x, **kwargs)  # test whether forward works
            self.tfms += [itf]

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Chained batch-wise transform for x.
        """
        for tf in self.tfms:
            x = tf.forward(x, **kwargs)
        return x

    def invert(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Chained batch-wise inverse transform for x.
        """
        for tf in reversed(self.tfms):
            x = tf.invert(x, **kwargs)
        return x.float()


def tfms_chain(
    tfms: List,
) -> Callable[[torch.Tensor, Dict[str, Any]], ContTransformerChain]:
    return partial(ContTransformerChain, tfms=tfms)


ContTransformerRangeExtended = partial(ContTransformerRange, x_range="-1-1")

ContTransformerLogRangeExtended = tfms_chain(
    [
        ContTransformerLog,
        partial(ContTransformerRange, x_range="-1-1"),
    ]
)

ContTransformerLogRangeGrouped = tfms_chain(
    [ContTransformerLog, ContTransformerRangeGrouped]
)
