# PLR Embeddings from https://github.com/yandex-research/rtdl-num-embeddings/ available via Apache 2.0 LICENSE
# Code was adapted per requirements
import math
from functools import partial
from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter


class _Periodic(nn.Module):
    """
    WARNING: the direct usage of this module is discouraged
    (do this only if you understand why this warning is here).
    """

    def __init__(self, n_features: int, k: int, sigma: float) -> None:
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, however: {sigma=}")

        super().__init__()
        self._sigma = sigma
        self.weight = Parameter(torch.empty(n_features, k))
        self.reset_parameters()

    def reset_parameters(self):
        # NOTE[DIFF]
        # Here, extreme values (~0.3% probability) are explicitly avoided just in case.
        # In the paper, there was no protection from extreme values.
        bound = self._sigma * 3
        nn.init.trunc_normal_(self.weight, 0.0, self._sigma, a=-bound, b=bound)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(
                f"The input must have at least two dimensions, however: {x.ndim=}"
            )

        x = 2 * math.pi * self.weight * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return x


# _NLinear is a simplified copy of delu.nn.NLinear:
# https://yura52.github.io/delu/stable/api/generated/delu.nn.NLinear.html
class _NLinear(nn.Module):
    """N *separate* linear layers for N feature embeddings."""

    def __init__(self, n: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(n, in_features, out_features))
        self.bias = Parameter(torch.empty(n, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        d_in_rsqrt = self.weight.shape[-2] ** -0.5
        nn.init.uniform_(self.weight, -d_in_rsqrt, d_in_rsqrt)
        nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3
        assert x.shape[-(self.weight.ndim - 1) :] == self.weight.shape[:-1]
        x = (x[..., None, :] @ self.weight).squeeze(-2)
        x = x + self.bias
        return x


class PeriodicEmbeddings(nn.Module):
    """PL & PLR & PLR(lite) (P ~ Periodic, L ~ Linear, R ~ ReLU) embeddings for continuous features.

    **Shape**

    - Input: `(*, n_features)`
    - Output: `(*, n_features, d_embedding)`

    **Examples**

    >>> batch_size = 2
    >>> n_cont_features = 3
    >>> x = torch.randn(batch_size, n_cont_features)
    >>>
    >>> # PLR embeddings (by default, d_embedding=24).
    >>> m = PeriodicEmbeddings(n_cont_features, lite=False)
    >>> m(x).shape
    torch.Size([2, 3, 24])
    >>>
    >>> # PLR(lite) embeddings.
    >>> m = PeriodicEmbeddings(n_cont_features, lite=True)
    >>> m(x).shape
    torch.Size([2, 3, 24])
    >>>
    >>> # PL embeddings.
    >>> m = PeriodicEmbeddings(n_cont_features, d_embedding=8, activation=False, lite=False)
    >>> m(x).shape
    torch.Size([2, 3, 8])
    """  # noqa: E501

    def __init__(
        self,
        n_features: int,
        d_embedding: int = 24,
        *,
        n_frequencies: int = 48,
        frequency_init_scale: float = 0.01,
        activation: bool = True,
        lite: bool,
    ) -> None:
        """
        Args:
            n_features: the number of features.
            d_embedding: the embedding size.
            n_frequencies: the number of frequencies for each feature.
                (denoted as "k" in Section 3.3 in the paper).
            frequency_init_scale: the initialization scale for the first linear layer
                (denoted as "sigma" in Section 3.3 in the paper).
                **This is an important hyperparameter**,
                see the documentation for details.
            activation: if True, the embeddings is PLR, otherwise, it is PL.
            lite: if True, the last linear layer (the "L" step)
                is shared between all features. See the README.md document for details.
        """
        super().__init__()
        self.periodic = _Periodic(n_features, n_frequencies, frequency_init_scale)
        self.linear: Union[nn.Linear, _NLinear]
        if lite:
            # NOTE[DIFF]
            # The PLR(lite) variation was not covered in this paper about embeddings,
            # but it was used in the paper about the TabR model.
            if not activation:
                raise ValueError("lite=True is allowed only when activation=True")
            self.linear = nn.Linear(2 * n_frequencies, d_embedding)
        else:
            self.linear = _NLinear(n_features, 2 * n_frequencies, d_embedding)
        self.activation = nn.ReLU() if activation else None

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        if x.ndim < 2:
            raise ValueError(
                f"The input must have at least two dimensions, however: {x.ndim=}"
            )

        x = self.periodic(x)
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        # x = F.flatten(x)
        # x = torch.flatten(x, 1)  # or adjust in _embed_features in models.py

        return x


if __name__ == "__main__":
    from yahpo_gym.benchmarks import iaml  # noqa: F401
    from yahpo_gym.configuration import cfg

    from yahpo_train.learner import SurrogateTabularLearner, dl_from_config
    from yahpo_train.losses import MultiMseLoss
    from yahpo_train.models import ResNet

    device = torch.device("cpu")

    cfg = cfg("iaml_glmnet")
    dl_train, dl_refit = dl_from_config(cfg, pin_memory=True, device=device)

    model = ResNet(
        dl_train,
        emb_plr=partial(PeriodicEmbeddings, lite=True),
        instance_names=cfg.instance_names,
    )
    surrogate = SurrogateTabularLearner(
        dl_train, model, loss_func=MultiMseLoss(), metrics=None
    )
    surrogate.fit_one_cycle(5, 1e-4)
