from yahpo_train.models_utils import *
from yahpo_train.models import AbstractSurrogate
from yahpo_train.learner import SurrogateTabularLearner
from torch.nn import Module, ModuleList


def sample_from_simplex(n, device):
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
    def __init__(self, base_model: AbstractSurrogate, n_models: int, **kwargs):
        super().__init__()
        self.models = ModuleList([base_model(**kwargs) for _ in range(n_models)])
        self.n_models = n_models

    def forward(self, x_cat, x_cont=None, invert_ytrafo=True) -> Tensor:
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
    `Ensemble Learner` for tabular data.
    """

    def __init__(self, dls, ensemble: Ensemble, dls_rng=1, **kwargs):
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
        n_epoch,
        lr_max=None,
        div=25.0,
        div_final=1e5,
        pct_start=0.25,
        wd=None,
        moms=None,
        cbs=None,
        reset_opt=False,
        start_epoch=0,
    ):
        for i in range(self.n_models):
            self._before_ens_fit(i)
            print(
                f"Training ensemble model {i + 1}/{self.n_models} with fit_one_cycle with lr {lr_max}"
            )
            self.learners[i].fit_one_cycle(
                n_epoch=np_epoch,
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
        n_epoch,
        lr=None,
        div_final=1e-5,
        pct_start=0.75,
        wd=None,
        cbs=None,
        reset_opt=False,
        start_epoch=0,
    ):
        for i in range(self.n_models):
            self._before_ens_fit(i)
            print(
                f"Training ensemble model {i + 1}/{self.n_models} with fit_flat_cos with lr {lr}"
            )
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

    def _before_ens_fit(self, i):
        # avoid duplicate callbacks
        self.learners[i].remove_cbs(self.learners[i].cbs)
        self.learners[i].add_cbs(self.cbs)
        # set RNG for dls
        self.dls.rng = self.dls_rng

    def _after_ens_fit(self, i):
        self.learners[i].remove_cbs(self.cbs)


if __name__ == "__main__":
    from yahpo_train.losses import *
    from yahpo_gym.configuration import cfg
    from yahpo_gym.benchmarks import iaml
    from yahpo_train.models import ResNet
    from yahpo_train.learner import SurrogateTabularLearner, dl_from_config

    device = torch.device("cpu")

    cfg = cfg("iaml_glmnet")
    dls = dl_from_config(cfg, pin_memory=True, device=device)

    ensemble = Ensemble(ResNet, n_models=3, dls=dls, instance_names=cfg.instance_names)
    surrogate = SurrogateEnsembleLearner(
        dls, ensemble, loss_func=MultiMaeLoss(), metrics=None
    )
    surrogate.fit_one_cycle(5, 1e-4)
