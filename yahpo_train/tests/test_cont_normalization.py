import torch
import numpy as np
from torch.functional import Tensor
from yahpo_train.cont_normalization import ContNormalization
from yahpo_gym.configuration import cfg
from yahpo_gym.benchmarks import lcbench
import pandas as pd
import scipy.stats
import pytest

def test_cont_norm():
    xss = [
        torch.rand(100,1),
        (torch.rand(1000,1) * 3) ** 2,
        -  5 * (torch.rand(500, 1) -.5),
        (torch.rand(500, 1) + 2) ** 3,
        torch.cat((torch.rand(500, 1), torch.rand(1,1) + torch.Tensor([1000.]))),
        torch.Tensor([1., 2., -1., 1000.]),
        torch.cat((torch.rand(800, 1), torch.rand(1,1) - torch.Tensor([1000.]))),
    ]
    xs2 = torch.cat((torch.rand(50, 1), torch.rand(1,1) + torch.Tensor([10000.]), torch.Tensor([-1000.]).unsqueeze(1)))
    for xs in xss:
        for normalize in ["scale", "range", None]:
            lim = 1e-3
            tfm = ContNormalization(xs, normalize = normalize,clip_outliers=False)
            xt = tfm(xs)
            _, lm1 = scipy.stats.yeojohnson(xs, lmbda=None)
            assert torch.abs(tfm.lmbda - lm1) < 1e-1
            xsn = tfm.invert(xt)            
            assert torch.mean(torch.abs(xsn - xs)) / torch.max(xs) < lim
            assert xsn.shape == xs.shape

            # New data
            xt2 = tfm(xs2)
            xsn2 = tfm.invert(xt2)
            assert torch.mean(torch.abs(xsn2 - xs2)) / torch.max(xs2) < lim
            assert xsn2.shape == xs2.shape

def test_cont_with_nan():
    xss = [
        torch.cat((torch.rand(50, 1), torch.Tensor(np.array([np.nan]).reshape(1,1)))),
        torch.Tensor([1., 2., -1., 1000, np.nan]),
        torch.Tensor([1., np.nan, -1., 1000, np.nan])
    ]
    for xs in xss:
        for normalize in ["scale", "range", None]:
            lim = 1e-4
            tfm = ContNormalization(xs, normalize = normalize)
            xt = tfm(xs)
            xsn = tfm.invert(xt)
            mask = ~torch.isnan(xs)
            diff = xsn - xs
            assert torch.mean(torch.abs(diff[mask])) / torch.max(xs[mask]) < lim
            assert xsn.shape == xs.shape

def test_cont_with_log():
    xss = [
        - torch.log(torch.rand(150, 1)),
        torch.float_power(torch.rand(150, 1)*2, 10.)
    ]
    for xs in xss:
        for normalize in ["scale", "range", None]:
            lim = 1e-4
            tfm = ContNormalization(xs, normalize = normalize)
            xt = tfm(xs)
            xsn = tfm.invert(xt)
            mask = ~torch.isnan(xs)
            diff = xsn - xs
            assert torch.mean(torch.abs(diff[mask])) / torch.max(xs[mask]) < lim
            assert xsn.shape == xs.shape

def test_cont_norm_pd():
    nrows = 1000000
    file = cfg("lcbench").get_path("dataset")
    df2 = pd.read_csv(file, nrows=nrows).sample(frac=.01)
    df = pd.read_csv(file, nrows=nrows).sample(frac=.3)
    for nm in df.columns[1:]:
        lim = 1e-3
        for normalize in ["scale", "range", None]:
            xs = torch.Tensor(df[nm].values)
            tfm = ContNormalization(xs, normalize = normalize)
            xt = tfm(xs)
            xsn = tfm.invert(xt)
            assert torch.mean(torch.abs(xsn - xs)) / torch.max(xs) < lim
            assert xsn.shape == xs.shape
            assert tfm.lmbda <= tfm.lmbda <= 5.1
            xs = torch.Tensor(df2[nm].values)
            xt = tfm(xs)
            xsn = tfm.invert(xt)
            assert torch.mean(torch.abs(xsn - xs)) / torch.max(xs) < lim
            assert xsn.shape == xs.shape

if __name__ == '__main__':
    test_cont_norm()
    test_cont_with_nan()
    test_cont_with_log()
    test_cont_norm_pd()
