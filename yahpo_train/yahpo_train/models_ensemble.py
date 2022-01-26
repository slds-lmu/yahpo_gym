import typing as ty

import torch
import torch.nn as nn
import torch.distributions as dist


from fastai.tabular.all import *

from yahpo_train.cont_scalers import *
from yahpo_train.models_utils import *
from yahpo_train.models import AbstractSurrogate
from yahpo_train.learner import SurrogateTabularLearner

def sample_from_simplex(n):
    """
    Sample from a simplex.
    Following https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
    """
    alpha = torch.rand(n+1)
    alpha[0] = 0.0
    alpha[-1] = 1.0
    alpha = torch.sort(alpha).values
    return alpha[1:] - alpha[:-1]

class Ensemble(AbstractSurrogate):
    def __init__(self, base_model: nn.Module, n_models: int, **kwargs):
        super().__init__()
        self.models = nn.ModuleList([base_model(**kwargs) for _ in range(n_models)])
        self.n_models = n_models

    def forward(self, x_cat, x_cont=None, invert_ytrafo=True) -> Tensor:
        ys = torch.stack([model(x_cat, x_cont, invert_ytrafo) for model in self.models], dim=0)
        # Draw ensemble weights
        alpha = sample_from_simplex(self.n_models).to(ys.device)
        
        # Compute weighted average
        ys = ys * alpha[:,None, None]
        return torch.sum(ys, dim=0)

class SurrogateEnsembleLearner(SurrogateTabularLearner):
    "`Ensemble Learner` for tabular data"

    def __init__(self, dls, ensemble: ty.List[nn.Module], **kwargs):
        super().__init__(dls, ensemble, **kwargs)
        self.learners = [SurrogateTabularLearner(dls, model, **kwargs) for model in ensemble.models]
        self.n_models = ensemble.n_models

    def fit_one_cycle(self, n_epoch, lr_max=None, div=25., div_final=1e5, pct_start=0.25, wd=None, moms=None, cbs=None, reset_opt=False):
        for i in range(self.n_models):
            print(f"Training ensemble model {i+1}/{self.n_models}")
            self.learners[i].fit_one_cycle(n_epoch, lr_max, div, div_final, pct_start, wd, moms, cbs, reset_opt)


if __name__ == '__main__':
    import torch.nn as nn
    from yahpo_gym.configuration import cfg
    from yahpo_gym.benchmarks import lcbench
    from yahpo_train.models import ResNet
    from yahpo_train.models_ensemble import *
    from yahpo_train.learner import SurrogateTabularLearner, dl_from_config

    cfg = cfg("lcbench")
    dls = dl_from_config(cfg, nrows=None)

    print('Resnet:')
    f = Ensemble(ResNet, n_models=3, dls=dls)
    l = SurrogateEnsembleLearner(dls, f, loss_func=nn.MSELoss(reduction='mean'), metrics=nn.MSELoss)
    l.add_cb(MixHandler)
    l.fit_one_cycle(1, 1e-4)
    l.export_onnx(cfg, 'cuda:0', suffix='noisy')



