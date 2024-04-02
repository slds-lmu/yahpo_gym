from typing import Callable

import numpy as np
import torch
from fastai.tabular.all import Metric
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

from yahpo_train.learner import SurrogateTabularLearner


class AvgTfedMetric(Metric):
    """
    Average the values of `func` taking into account potential different batch sizes.
    Specializes to transformed metrics saved during get_one_batch
    """

    def __init__(self, func: Callable):
        self.func = func
        self.total = 0.0
        self.count = 0

    def reset(self) -> None:
        """
        Reset the metric.
        """
        self.total, self.count = 0.0, 0

    def accumulate(self, learner: SurrogateTabularLearner) -> None:
        """
        Accumulate the predictions and the targets.
        """
        bs = find_bs(learner.tfyb)
        if torch.isnan(learner.tfpred).any():
            raise ValueError("NaNs in predictions.")
        self.total += (
            learner.to_detach(self.func(*learner.tfyb, learner.tfpred)) * bs
        )  # func is called with truth and prediction
        self.count += bs

    @property
    def value(self) -> float:
        """
        Value of the metric.
        """
        return self.total / self.count if self.count != 0 else None

    @property
    def name(self) -> str:
        """
        Name of the metric.
        """
        return (
            self.func.func.__name__
            if hasattr(self.func, "func")
            else self.func.__name__
        )


def mae(x: torch.Tensor, y: torch.Tensor, impute_nan: bool = True) -> np.ndarray:
    """
    MAE
    """
    if impute_nan:
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)
    return mean_absolute_error(x.cpu(), y_pred=y.cpu(), multioutput="raw_values")


def r2(x: torch.Tensor, y: torch.Tensor, impute_nan: bool = True) -> np.ndarray:
    """R2"""
    if impute_nan:
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)
    return r2_score(x.cpu(), y_pred=y.cpu(), multioutput="raw_values")


def spearman(x: torch.Tensor, y: torch.Tensor, impute_nan: bool = True) -> np.ndarray:
    """
    Spearman
    """
    if impute_nan:
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)

    # Return 0 for constant batches
    if torch.all(y == y[1]) or torch.all(x == x[1]):
        x = np.array(x.cpu())
        y = np.array(y.cpu())
        return np.array([0 for _, _ in zip(np.rollaxis(x, 1), np.rollaxis(y, 1))])

    x = np.array(x.cpu())
    y = np.array(y.cpu())

    rho = [
        spearmanr(xs, b=ys)[0] if not (all(xs[0] == xs) or all(ys[0] == ys)) else 0.0
        for xs, ys in zip(np.rollaxis(x, 1), np.rollaxis(y, 1))
    ]
    return np.array(rho)


def pearson(x: torch.Tensor, y: torch.Tensor, impute_nan: bool = True) -> np.ndarray:
    """
    Pearson
    """
    if impute_nan:
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)

    # Return 0 for constant batches
    if torch.all(y == y[1]) or torch.all(x == x[1]):
        x = np.array(x.cpu())
        y = np.array(y.cpu())
        return np.array([0 for _, _ in zip(np.rollaxis(x, 1), np.rollaxis(y, 1))])

    x = np.array(x.cpu())
    y = np.array(y.cpu())

    r = [
        pearsonr(xs, y=ys)[0] if not (all(xs[0] == xs) or all(ys[0] == ys)) else 0.0
        for xs, ys in zip(np.rollaxis(x, 1), np.rollaxis(y, 1))
    ]
    return np.array(r)


def napct(x: torch.Tensor, y: torch.Tensor, impute_nan: bool = True) -> np.ndarray:
    """
    Percentage of NaNs
    """
    return torch.mean(torch.isnan(y.cpu()).float()).numpy()
