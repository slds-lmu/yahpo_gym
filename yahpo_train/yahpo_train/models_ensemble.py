from typing import Callable, List, Optional, Tuple, Type, Union

from fastai.callback.core import Callback
from fastai.tabular.data import TabularDataLoaders
from torch.nn import Module, ModuleList

from yahpo_train.learner import SurrogateTabularLearner
from yahpo_train.models import AbstractSurrogate
from yahpo_train.models_utils import *


def sample_from_simplex(n: int, device: torch.device) -> torch.Tensor:
    """
    Sample from a simplex.
    Following https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
    """
    alpha = torch.rand(n + 1).to(device=device)
    alpha[0] = 0.0
    alpha[n] = 1.0
    alpha = torch.sort(alpha).values
    return alpha[1:] - alpha[:n]


class Ensemble(AbstractSurrogate):
    """
    Abstract class for ensemble models.
    """

    def __init__(self, base_model: Type[AbstractSurrogate], n_models: int, **kwargs):
        super().__init__()
        self.models = ModuleList([base_model(**kwargs) for _ in range(n_models)])
        self.n_models = n_models

    def forward(
        self, x_cat: torch.Tensor, x_cont: torch.Tensor, invert_ytrafo: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through the ensemble.
        """
        ys = torch.stack(
            [model(x_cat, x_cont, invert_ytrafo) for model in self.models], dim=0
        )
        # draw ensemble weights
        alpha = sample_from_simplex(self.n_models, ys.device)
        # compute weighted average
        ys = ys * alpha[:, None, None]
        return torch.sum(ys, dim=0)


class SurrogateEnsembleLearner(SurrogateTabularLearner):
    """
    Learner for ensemble models.
    """

    def __init__(
        self,
        dls: TabularDataLoaders,
        ensemble: Ensemble,
        dls_rng: int = 1,
        **kwargs,
    ):
        self.learners = [
            SurrogateTabularLearner(dls, model, **kwargs) for model in ensemble.models
        ]
        # remove all cbs because they are added and removed if needed during fits from the super class, see below
        for i in range(0, len(self.learners)):
            self.learners[i].remove_cbs(self.learners[i].cbs)
        self.n_models = ensemble.n_models

        if dls_rng is not None:
            self.dls_rng = dls_rng

        super().__init__(dls, ensemble, **kwargs)

    def fit_one_cycle(
        self,
        n_epoch: int,
        lr_max: Optional[float] = None,
        div: float = 25.0,
        div_final: float = 1e5,
        pct_start: float = 0.25,
        wd: Optional[float] = None,
        moms: Optional[Tuple[float, float]] = None,
        cbs: Optional[Union[Callback, List[Callback]]] = None,
        reset_opt: bool = False,
        start_epoch: int = 0,
    ) -> None:
        """
        Fit each `self.learner` for `n_epoch` using the 1cycle policy.
        """
        for i in range(self.n_models):
            self._before_ens_fit(i)
            print(
                f"Training ensemble model {i + 1}/{self.n_models} with fit_one_cycle with lr {lr_max}"
            )
            with self.learners[i].no_bar():  # prevent duplicate progress bars
                self.learners[i].fit_one_cycle(
                    n_epoch=n_epoch,
                    lr_max=lr_max,
                    div=div,
                    div_final=div_final,
                    pct_start=pct_start,
                    wd=wd,
                    moms=moms,
                    cbs=cbs,
                    reset_opt=reset_opt,
                    start_epoch=start_epoch,
                )
            self._after_ens_fit(i)

    def fit_flat_cos(
        self,
        n_epoch: int,
        lr: Optional[float] = None,
        div_final: float = 1e-5,
        pct_start: float = 0.75,
        wd: Optional[float] = None,
        cbs: Optional[Union[Callback, List[Callback]]] = None,
        reset_opt: bool = False,
        start_epoch: int = 0,
    ) -> None:
        """
        Fit each `self.learner` for `n_epoch` using the flat cos policy.
        """
        for i in range(self.n_models):
            self._before_ens_fit(i)
            print(
                f"Training ensemble model {i + 1}/{self.n_models} with fit_flat_cos with lr {lr}"
            )
            with self.learners[i].no_bar():  # prevent duplicate progress bars
                self.learners[i].fit_flat_cos(
                    n_epoch=n_epoch,
                    lr=lr,
                    div_final=div_final,
                    pct_start=pct_start,
                    wd=wd,
                    cbs=cbs,
                    reset_opt=reset_opt,
                    start_epoch=start_epoch,
                )
            self._after_ens_fit(i)

    def _before_ens_fit(self, i: int) -> None:
        """
        Add callbacks to the learner and set the RNG for the dataloader.
        """
        # avoid duplicate callbacks
        self.learners[i].remove_cbs(self.learners[i].cbs)
        self.learners[i].add_cbs(self.cbs)
        # set RNG for dls
        self.dls.rng = self.dls_rng

    def _after_ens_fit(self, i: int) -> None:
        """
        Remove callbacks from the learner.
        """
        self.learners[i].remove_cbs(self.cbs)


if __name__ == "__main__":
    from yahpo_gym.benchmarks import iaml
    from yahpo_gym.configuration import cfg

    from yahpo_train.learner import SurrogateTabularLearner, dl_from_config
    from yahpo_train.losses import *
    from yahpo_train.models import ResNet

    device = torch.device("cpu")

    cfg = cfg("iaml_glmnet")
    dl_train, dl_refit = dl_from_config(cfg, pin_memory=True, device=device)

    ensemble = Ensemble(
        ResNet, n_models=3, dls=dl_train, instance_names=cfg.instance_names
    )
    surrogate = SurrogateEnsembleLearner(
        dl_train, ensemble, loss_func=MultiMseLoss(), metrics=None
    )
    surrogate.fit_one_cycle(5, 1e-4)
