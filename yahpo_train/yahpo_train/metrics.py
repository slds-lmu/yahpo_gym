from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
from fastai.tabular.all import *


class AvgTfedMetric(Metric):
    """
    Average the values of `func` taking into account potential different batch sizes.
    Specializes to transformed metrics saved during get_one_batch
    """

    def __init__(self, func):
        self.func = func
        self.total = 0.0
        self.count = 0

    def reset(self):
        self.total, self.count = 0.0, 0

    def accumulate(self, learn):
        bs = find_bs(learn.tfyb)
        self.total += (
            learn.to_detach(self.func(*learn.tfyb, learn.tfpred)) * bs
        )  # func is called with truth and pred
        self.count += bs

    @property
    def value(self):
        return self.total / self.count if self.count != 0 else None

    @property
    def name(self):
        return (
            self.func.func.__name__
            if hasattr(self.func, "func")
            else self.func.__name__
        )


def mae(x, y, impute_nan=True):
    if impute_nan:
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)
    return mean_absolute_error(x.cpu(), y_pred=y.cpu(), multioutput="raw_values")


def r2(x, y, impute_nan=True):
    if impute_nan:
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)
    return r2_score(x.cpu(), y_pred=y.cpu(), multioutput="raw_values")


def spearman(x, y, impute_nan=True):
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
        spearmanr(xs, b=ys)[0]
        if not ((xs[0] == xs).all() or (ys[0] == ys).all())
        else 0.0
        for xs, ys in zip(np.rollaxis(x, 1), np.rollaxis(y, 1))
    ]
    return np.array(rho)


def pearson(x, y, impute_nan=True):
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
        pearsonr(xs, y=ys)[0]
        if not ((xs[0] == xs).all() or (ys[0] == ys).all())
        else 0.0
        for xs, ys in zip(np.rollaxis(x, 1), np.rollaxis(y, 1))
    ]
    return np.array(r)


def napct(x, y, impute_nan=True):
    return torch.mean(torch.isnan(y).float())
