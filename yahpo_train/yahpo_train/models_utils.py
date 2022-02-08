import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.onnx
import typing as ty

# Repurposed and adapted from https://github.com/yandex-research/rtdl under Apache License 2.0
def reglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)

def geglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)

class ReGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)

def get_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
    return (
        reglu
        if name == 'reglu'
        else geglu
        if name == 'geglu'
        else torch.sigmoid
        if name == 'sigmoid'
        else getattr(F, name)
    )

def get_nonglu_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
    return (
        F.relu
        if name == 'reglu'
        else F.gelu
        if name == 'geglu'
        else get_activation_fn(name)
    )


# The following functions are adapted from https://github.com/fastai/fastai and changed according to new requirements
from fastai.tabular.all import ifnone
from fastai.torch_basics import *
from fastai.tabular.all import *

def emb_sz_rule(n_cat):
    "Rule of thumb to pick embedding size corresponding to `n_cat`"
    return min(600, round(1.6 * n_cat**0.56)*2)

def _one_emb_sz(classes, n, sz_dict=None):
    "Pick an embedding size for `n` depending on `classes` if not given in `sz_dict`."
    sz_dict = ifnone(sz_dict, {})
    n_cat = len(classes[n])
    sz = sz_dict.get(n, int(emb_sz_rule(n_cat)))
    return n_cat,sz

def get_emb_sz(to, sz_dict=None):
    "Get default embedding size from `TabularPreprocessor` `proc` or the ones in `sz_dict`"
    return [_one_emb_sz(to.classes, n, sz_dict) for n in to.cat_names]
