import torch
import numpy as np
from yahpo_train.cont_scalers import *

def test_cont_scaler():
    x = torch.rand(100)

    cr = ContTransformerLogRange(x)
    x_tf = cr(x)
    x_rec = cr.invert(x_tf)

    assert torch.max(torch.abs(x_rec - x)).numpy() < 1e-5

if __name__ == '__main__':
    test_cont_scaler()

