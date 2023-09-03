import argparse
import logging
import random
import warnings
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import optuna
import torch
from fastai.tabular.data import TabularDataLoaders
from optuna.integration import FastAIPruningCallback
from optuna.study import Study
from yahpo_gym.configuration import cfg

from yahpo_train.cont_scalers import *
from yahpo_train.helpers import generate_all_test_set_metrics
from yahpo_train.learner import *
from yahpo_train.losses import *
from yahpo_train.metrics import *
from yahpo_train.models import *
from yahpo_train.models_ensemble import *


def random_seed(seed: int, use_cuda: bool) -> None:
    np.random.seed(seed)  # cpu vars
    torch.manual_seed(seed)  # cpu  vars
    random.seed(seed)  # python
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def fit_config_resnet(
    key: str,
    dl_train: TabularDataLoaders,
    noisy: bool = False,
    embds_dbl: Optional[Union[List[nn.Module], List[functools.partial]]] = None,
    embds_tgt: Optional[Union[List[nn.Module], List[functools.partial]]] = None,
    tfms: Optional[Dict[str, Callable]] = None,
    lr: float = 1e-4,
    wd: Optional[float] = None,
    moms: Optional[Tuple[float, float]] = None,
    epochs: int = 50,
    d: int = 256,
    d_hidden_factor: float = 2.0,
    n_layers: int = 4,
    hidden_dropout: float = 0.0,
    residual_dropout: float = 0.2,
    fit_cbs: Optional[List[Callback]] = [],
) -> SurrogateTabularLearner:
    """
    Fit function with hyperparameters for resnet.
    """
    config = cfg(key)

    # construct embds from tfms
    # tfms overwrites emdbs_dbl, embds_tgt
    if tfms is not None:
        embds_dbl = [
            tfms.get(name)
            if tfms.get(name) is not None
            else ContTransformerRangeExtended
            for name, cont in dl_train.all_cols[dl_train.cont_names].items()
        ]
        embds_tgt = [
            tfms.get(name)
            if tfms.get(name) is not None
            else (
                ContTransformerRangeGrouped
                if config.instance_names is not None
                else ContTransformerRange
            )
            for name, cont in dl_train.ys.items()
        ]

    # instantiate learner
    if noisy:
        model = Ensemble(
            ResNet,
            n_models=3,
            dls=dl_train,
            embds_dbl=embds_dbl,
            embds_tgt=embds_tgt,
            instance_names=config.instance_names,
            d=d,
            d_hidden_factor=d_hidden_factor,
            n_layers=n_layers,
            hidden_dropout=hidden_dropout,
            residual_dropout=residual_dropout,
        )
        surrogate = SurrogateEnsembleLearner(
            dl_train, ensemble=model, loss_func=MultiMseLoss()
        )
        # NOTE: this is ugly, we probably should overload the metric setter and getter for the SurrogateEnsembleLearner
        surrogate.metrics = [
            AvgTfedMetric(mae),
            AvgTfedMetric(r2),
            AvgTfedMetric(spearman),
            AvgTfedMetric(pearson),
            AvgTfedMetric(napct),
        ]
        for i in range(len(surrogate.learners)):
            surrogate.learners[i].metrics = [
                AvgTfedMetric(mae),
                AvgTfedMetric(r2),
                AvgTfedMetric(spearman),
                AvgTfedMetric(pearson),
                AvgTfedMetric(napct),
            ]
    else:
        model = ResNet(
            dl_train,
            embds_dbl=embds_dbl,
            embds_tgt=embds_tgt,
            instance_names=config.instance_names,
            d=d,
            d_hidden_factor=d_hidden_factor,
            n_layers=n_layers,
            hidden_dropout=hidden_dropout,
            residual_dropout=residual_dropout,
        )
        surrogate = SurrogateTabularLearner(
            dl_train, model=model, loss_func=MultiMseLoss()
        )
        surrogate.metrics = [
            AvgTfedMetric(mae),
            AvgTfedMetric(r2),
            AvgTfedMetric(spearman),
            AvgTfedMetric(pearson),
            AvgTfedMetric(napct),
        ]

    # fit
    if moms[0] < moms[1]:
        moms = (moms[1], moms[0], moms[1])
    else:
        moms = (moms[0], moms[1], moms[0])
    surrogate.fit_one_cycle(epochs, lr_max=lr, wd=wd, moms=moms, cbs=fit_cbs)

    return surrogate


def tune_config_resnet(
    key: str,
    name: str,
    dl_train: TabularDataLoaders,
    tfms_fixed: Dict[str, Callable] = {},
    trials: int = 0,
    walltime: float = 0,
    **kwargs
) -> optuna.study.Study:
    if trials == 0:
        trials = None

    if walltime == 0:
        walltime = None

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    storage = "sqlite:///{}.db".format(Path(cfg(key).config_path, name))
    sampler = optuna.samplers.TPESampler(seed=10, n_ei_candidates=1000)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)
    study = optuna.create_study(
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        study_name=name,
        direction="minimize",
        load_if_exists=True,
    )

    # for the search space see https://arxiv.org/pdf/2106.11959.pdf
    # except for wd and moms
    def objective(trial):
        d = trial.suggest_int("d", 64, 512, step=64)  # layer size
        d_hidden_factor = trial.suggest_float(
            "d_hidden_factor", 1.0, 4.0
        )  # hidden factor
        n_layers = trial.suggest_int("n_layers", 1, 8)  # number of layers
        hidden_dropout = trial.suggest_float(
            "hidden_dropout", 0.0, 0.5
        )  # hidden dropout
        use_residual_dropout = trial.suggest_categorical(
            "use_residual_dropout", [True, False]
        )
        if use_residual_dropout:
            residual_dropout = trial.suggest_float("residual_dropout", 1e-2, 0.5)
        else:
            residual_dropout = 0.0
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        use_wd = trial.suggest_categorical("use_wd", [True, False])
        if use_wd:
            wd = trial.suggest_float("wd", 1e-7, 1e-2, log=True)
        else:
            wd = 0.0
        mom1 = trial.suggest_float("mom1", 0.85, 0.99)
        mom2 = trial.suggest_float("mom2", 0.85, 0.99)
        moms = (mom1, mom2)
        cbs = [FastAIPruningCallback(trial=trial, monitor="valid_loss")]

        surrogate = fit_config_resnet(
            key=key,
            dl_train=dl_train,
            tfms=tfms_fixed,
            lr=lr,
            wd=wd,
            moms=moms,
            d=d,
            d_hidden_factor=d_hidden_factor,
            n_layers=n_layers,
            hidden_dropout=hidden_dropout,
            residual_dropout=residual_dropout,
            fit_cbs=cbs,
            **kwargs,
        )
        loss = surrogate.validate()[0]
        return loss

    study.optimize(objective, n_trials=trials, timeout=walltime)
    return study


if __name__ == "__main__":
    tfms_list = {}

    tfms_lcbench = {}
    tfms_lcbench.update(
        {
            "val_accuracy": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.00, max=1.00),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "val_cross_entropy": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.00),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "val_balanced_accuracy": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.00, max=1.00),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "test_accuracy": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.00, max=1.00),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "test_cross_entropy": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.00),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "test_balanced_accuracy": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.00, max=1.00),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "time": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.00),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "time_increase": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.00),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "model_parameters": tfms_chain(
                [
                    partial(ContTransformerClamp, min=1.00),
                    ContTransformerInt,
                    ContTransformerRangeBoxCoxStandardizeRange,
                ]
            ),
            "batch_size": ContTransformerLogRangeExtended,
            "learning_rate": ContTransformerLogRangeExtended,
            "max_units": ContTransformerLogRangeExtended,
        }
    )
    tfms_list.update({"lcbench": tfms_lcbench})

    tfms_nb301 = {}
    tfms_nb301.update(
        {
            "val_accuracy": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.00, max=1.00),
                    ContTransformerRangeBoxCoxStandardizeRange,
                ]
            ),
            "val_cross_entropy": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.00),
                    ContTransformerRangeBoxCoxStandardizeRange,
                ]
            ),
            "runtime": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.00),
                    ContTransformerRangeBoxCoxStandardizeRange,
                ]
            ),
            "model_parameters": tfms_chain(
                [
                    partial(ContTransformerClamp, min=1.00),
                    ContTransformerInt,
                    ContTransformerRangeBoxCoxStandardizeRange,
                ]
            ),
        }
    )
    tfms_list.update({"nb301": tfms_nb301})

    tfms_rbv2 = {}
    tfms_rbv2.update(
        {
            "acc": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.00, max=1.00),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "bac": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.00, max=1.00),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "auc": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.00, max=1.00),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "brier": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.00, max=2.00),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "logloss": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.00),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "timetrain": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.00),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "timepredict": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.00),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
        }
    )

    tfms_rbv2_glmnet = tfms_rbv2.copy()
    tfms_rbv2_glmnet.update(
        {
            "s": ContTransformerLogRangeExtended,
        }
    )

    tfms_rbv2_rpart = tfms_rbv2.copy()
    tfms_rbv2_rpart.update(
        {
            "cp": ContTransformerLogRangeExtended,
        }
    )

    tfms_rbv2_aknn = tfms_rbv2.copy()
    tfms_rbv2_aknn.update(
        {
            "ef": ContTransformerLogRangeExtended,
            "ef_construction": ContTransformerLogRangeExtended,
        }
    )

    tfms_rbv2_svm = tfms_rbv2.copy()
    tfms_rbv2_svm.update(
        {
            "cost": ContTransformerShiftLogRangeExtended,
            "gamma": ContTransformerShiftLogRangeExtended,
            "tolerance": ContTransformerShiftLogRangeExtended,
        }
    )

    tfms_rbv2_ranger = tfms_rbv2.copy()

    tfms_rbv2_xgboost = tfms_rbv2.copy()
    tfms_rbv2_xgboost.update(
        {
            "nrounds": ContTransformerLogRangeExtended,
            "eta": ContTransformerLogRangeExtended,
            "gamma": ContTransformerLogRangeExtended,
            "lambda": ContTransformerLogRangeExtended,
            "alpha": ContTransformerLogRangeExtended,
            "min_child_weight": ContTransformerLogRangeExtended,
        }
    )

    tfms_rbv2_super = tfms_rbv2.copy()
    tfms_rbv2_super.update(
        {
            "glmnet.s": ContTransformerLogRangeExtended,
            "rpart.cp": ContTransformerLogRangeExtended,
            "aknn.ef": ContTransformerLogRangeExtended,
            "aknn.ef_construction": ContTransformerLogRangeExtended,
            "svm.cost": ContTransformerShiftLogRangeExtended,
            "svm.gamma": ContTransformerShiftLogRangeExtended,
            "svm.tolerance": ContTransformerShiftLogRangeExtended,
            "xgboost.nrounds": ContTransformerLogRangeExtended,
            "xgboost.eta": ContTransformerLogRangeExtended,
            "xgboost.gamma": ContTransformerLogRangeExtended,
            "xgboost.lambda": ContTransformerLogRangeExtended,
            "xgboost.alpha": ContTransformerLogRangeExtended,
            "xgboost.min_child_weight": ContTransformerLogRangeExtended,
        }
    )

    tfms_list.update(
        {
            "rbv2_glmnet": tfms_rbv2_glmnet,
            "rbv2_rpart": tfms_rbv2_rpart,
            "rbv2_aknn": tfms_rbv2_aknn,
            "rbv2_svm": tfms_rbv2_svm,
            "rbv2_ranger": tfms_rbv2_ranger,
            "rbv2_xgboost": tfms_rbv2_xgboost,
            "rbv2_super": tfms_rbv2_super,
        }
    )

    tfms_iaml = {}
    tfms_iaml.update(
        {
            "mmce": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=1.0),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "f1": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=1.0),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "auc": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=1.0),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "logloss": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "rammodel": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "timetrain": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "mec": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "ias": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "nf": tfms_chain(
                [
                    partial(
                        ContTransformerClampGrouped,
                        min=[0, 0, 0, 0],  # 1067, 1489, 40981, 41146
                        max=[21, 5, 14, 20],  # 1067, 1489, 40981, 41146
                    ),
                    ContTransformerInt,
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
        }
    )

    tfms_iaml_glmnet = tfms_iaml.copy()
    tfms_iaml_glmnet.update(
        {
            "s": ContTransformerLogRangeExtended,
        }
    )

    tfms_iaml_rpart = tfms_iaml.copy()
    tfms_iaml_rpart.update(
        {
            "cp": ContTransformerLogRangeExtended,
        }
    )

    tfms_iaml_ranger = tfms_iaml.copy()

    tfms_iaml_xgboost = tfms_iaml.copy()
    tfms_iaml_xgboost.update(
        {
            "nrounds": ContTransformerLogRangeExtended,
            "eta": ContTransformerLogRangeExtended,
            "gamma": ContTransformerLogRangeExtended,
            "lambda": ContTransformerLogRangeExtended,
            "alpha": ContTransformerLogRangeExtended,
            "min_child_weight": ContTransformerLogRangeExtended,
        }
    )

    tfms_iaml_super = tfms_iaml.copy()
    tfms_iaml_super.update(
        {
            "glmnet.s": ContTransformerLogRangeExtended,
            "rpart.cp": ContTransformerLogRangeExtended,
            "xgboost.nrounds": ContTransformerLogRangeExtended,
            "xgboost.eta": ContTransformerLogRangeExtended,
            "xgboost.gamma": ContTransformerLogRangeExtended,
            "xgboost.lambda": ContTransformerLogRangeExtended,
            "xgboost.alpha": ContTransformerLogRangeExtended,
            "xgboost.min_child_weight": ContTransformerLogRangeExtended,
        }
    )

    tfms_list.update(
        {
            "iaml_glmnet": tfms_iaml_glmnet,
            "iaml_rpart": tfms_iaml_rpart,
            "iaml_ranger": tfms_iaml_ranger,
            "iaml_xgboost": tfms_iaml_xgboost,
            "iaml_super": tfms_iaml_super,
        }
    )

    tfms_fair = {}
    tfms_fair.update(
        {
            "mmce": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=1.0),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "f1": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=1.0),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "feo": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=1.0),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "facc": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=1.0),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "ftpr": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=1.0),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "ffomr": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=1.0),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "ffnr": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=1.0),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "rammodel": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
            "timetrain": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0),
                    ContTransformerRangeGroupedBoxCoxGroupedStandardizeGroupedRangeGrouped,
                ]
            ),
        }
    )

    tfms_fair_fgrrm = tfms_fair.copy()
    tfms_fair_fgrrm.update(
        {
            "lambda": ContTransformerLogRangeExtended,
        }
    )

    tfms_fair_rpart = tfms_fair.copy()
    tfms_fair_rpart.update(
        {
            "cp": ContTransformerLogRangeExtended,
        }
    )

    tfms_fair_ranger = tfms_fair.copy()

    tfms_fair_xgboost = tfms_fair.copy()
    tfms_fair_xgboost.update(
        {
            "nrounds": ContTransformerLogRangeExtended,
            "eta": ContTransformerLogRangeExtended,
            "gamma": ContTransformerLogRangeExtended,
            "lambda": ContTransformerLogRangeExtended,
            "alpha": ContTransformerLogRangeExtended,
            "min_child_weight": ContTransformerLogRangeExtended,
        }
    )

    tfms_fair_super = tfms_fair.copy()
    tfms_fair_super.update(
        {
            "fgrrm.lambda": ContTransformerLogRangeExtended,
            "rpart.cp": ContTransformerLogRangeExtended,
            "xgboost.nrounds": ContTransformerLogRangeExtended,
            "xgboost.eta": ContTransformerLogRangeExtended,
            "xgboost.gamma": ContTransformerLogRangeExtended,
            "xgboost.lambda": ContTransformerLogRangeExtended,
            "xgboost.alpha": ContTransformerLogRangeExtended,
            "xgboost.min_child_weight": ContTransformerLogRangeExtended,
        }
    )

    tfms_list.update(
        {
            "fair_fgrrm": tfms_fair_fgrrm,
            "fair_rpart": tfms_fair_rpart,
            "fair_ranger": tfms_fair_ranger,
            "fair_xgboost": tfms_fair_xgboost,
            "fair_super": tfms_fair_super,
        }
    )

    parser = argparse.ArgumentParser(description="Args for resnet tuning")
    parser.add_argument(
        "--key",
        type=str,
        default="iaml_glmnet
        help="Key of benchmark scenario, e.g., 'iaml_glmnet'",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="tune_iaml_glmnet_resnet",
        help="Name of the optuna study, e.g., 'tune_iaml_glmnet_resnet'",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=0,
        help="Number of optuna trials",
    )  # by default we run until terminated externally
    parser.add_argument(
        "--walltime",
        type=int,
        default=0,
        help="Walltime for optuna timeout in seconds",
    )  # by default we run until terminated externally
    args = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name()
        device = torch.device("cuda:0")
        use_cuda = True
        print("Using cuda device: " + device_name + " " + str(current_device))
    else:
        warnings.warn(
            "No cuda device available. You probably do not want to tune on CPUs."
        )
        device = torch.device("cpu")
        use_cuda = False

    config = cfg(args.key)
    dl_train, dl_refit = dl_from_config(
        config,
        save_df_test=True,
        save_encoding=True,
        shuffle=True,
        device=device,
        pin_memory=True,
    )

    random_seed(10, use_cuda=use_cuda)

    study = tune_config_resnet(
        args.key,
        name=args.name,
        dl_train=dl_train,
        tfms_fixed=tfms_list.get(args.key),
        trials=args.trials,
        walltime=args.walltime,
    )

    best_params = study.best_params
    with open(Path(config.config_path, config.config.get("best_params")), "w") as f:
        json.dump(best_params, f)

    # fix some parameters
    if not best_params.get("use_residual_dropout"):
        best_params.update({"residual_dropout": 0})

    if not best_params.get("use_wd"):
        best_params.update({"wd": 0})

    best_params.pop("use_residual_dropout")
    best_params.pop("use_wd")

    mom1 = best_params.get("mom1")
    mom2 = best_params.get("mom2")
    moms = (mom1, mom2)
    best_params.update({"moms": moms})
    best_params.pop("mom1")
    best_params.pop("mom2")

    warnings.filterwarnings(
        "ignore", category=UserWarning
    )  # ignore warnings due to empty validation set
    surrogate = fit_config_resnet(
        args.key,
        dl_train=dl_refit,
        tfms=tfms_list.get(args.key),
        **best_params,
    )
    warnings.filterwarnings("default", category=UserWarning)  # reset warnings

    surrogate.export_onnx(config, device=device)
    generate_all_test_set_metrics(
        args.key, model=config.config.get("model"), save_to_csv=True
    )

    # warnings.filterwarnings(
    #    "ignore", category=UserWarning
    # )  # ignore warnings due to empty validation set
    # surrogate_noisy = fit_config_resnet(
    #    args.key,
    #    dl_train=dl_refit,
    #    tfms=tfms_list.get(args.key),
    #    **best_params,
    #    noisy=True,
    # )
    # warnings.filterwarnings("default", category=UserWarning)  # reset warnings

    # surrogate_noisy.export_onnx(config, device=device, suffix="noisy")
