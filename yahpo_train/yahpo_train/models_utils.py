import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from typing import Callable
from fastai.tabular.all import *
from fastai.torch_basics import *


# Repurposed and adapted from https://github.com/yandex-research/rtdl under Apache License 2.0
def reglu(x: torch.Tensor) -> torch.Tensor:
    """
    ReGLU activation function.
    """
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: torch.Tensor) -> torch.Tensor:
    """
    GeGLU activation function.
    """
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class ReGLU(nn.Module):
    """
    ReGLU activation function class.
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ReGLU activation function.
        """
        return reglu(x)


class GeGLU(nn.Module):
    """
    GeGLU activation function class.
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GeGLU activation function.
        """
        return geglu(x)


def get_activation_fn(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Get activation function by name.
    """
    return (
        reglu
        if name == "reglu"
        else (
            geglu
            if name == "geglu"
            else torch.sigmoid if name == "sigmoid" else getattr(F, name)
        )
    )


def get_nonglu_activation_fn(
    name: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Get activation function by name.
    """
    return (
        F.relu
        if name == "reglu"
        else F.gelu if name == "geglu" else get_activation_fn(name)
    )


# The following functions are adapted from https://github.com/fastai/fastai and changed according to new requirements
def emb_sz_rule(n_cat: int) -> int:
    """
    Rule of thumb to pick embedding size corresponding to `n_cat`.
    """
    return min(600, round(1.6 * n_cat**0.56) * 2)


def _one_emb_sz(
    classes: Dict[str, List[str]], n: str, sz_dict: Optional[Dict] = None
) -> Tuple[int, int]:
    """
    Pick an embedding size for `n` depending on `classes` if not given in `sz_dict`.
    """
    sz_dict = ifnone(sz_dict, {})
    n_cat = len(classes[n])
    sz = sz_dict.get(n, int(emb_sz_rule(n_cat)))
    return n_cat, sz


def get_emb_sz(
    to: TabularPandas, sz_dict: Optional[Dict[str, int]] = None
) -> List[Tuple[int, int]]:
    """
    Get default embedding size from `TabularPreprocessor` `proc` or the ones in `sz_dict`.
    """
    return [_one_emb_sz(to.classes, n, sz_dict) for n in to.cat_names]
