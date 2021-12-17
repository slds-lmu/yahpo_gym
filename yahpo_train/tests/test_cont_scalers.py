import pytest
import torch
import numpy as np
from yahpo_train.cont_scalers import *

def test_cont_scaler():
    def cont_scaler_helper(x, transformer, eps = 1e-5):
        cr = transformer(x)
        x_tf = cr(x)
        x_rec = cr.invert(x_tf)
        assert torch.max(torch.abs(x_rec - x)).numpy() < eps

    x = torch.rand(100)

    for transformer in [ContTransformerNone, ContTransformerRange, ContTransformerNegExp, ContTransformerLog, ContTransformerMultScalar]:
        cont_scaler_helper(x, transformer)

    for transformer in [ContTransformerNone, ContTransformerRange, ContTransformerNegExp, ContTransformerMultScalar]:
        cont_scaler_helper(-x, transformer)

if __name__ == '__main__':
    test_cont_scaler()

