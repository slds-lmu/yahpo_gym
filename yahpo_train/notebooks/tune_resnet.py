from yahpo_train.models import *
from yahpo_train.models_ensemble import *
from yahpo_train.learner import *
from yahpo_train.losses import *
from yahpo_train.metrics import *
from yahpo_train.cont_scalers import *
from yahpo_gym.configuration import cfg
from yahpo_train.helpers import generate_all_test_set_metrics
import argparse
import optuna
from optuna.integration import FastAIPruningCallback
import torch
import random
import numpy as np
import logging
import warnings


def random_seed(seed, use_cuda):
    np.random.seed(seed)  # cpu vars
    torch.manual_seed(seed)  # cpu  vars
    random.seed(seed)  # python
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def fit_config_resnet(
    key,
    dl_train,
    noisy=False,
    embds_dbl=None,
    embds_tgt=None,
    tfms=None,
    fit="fit_flat_cos",
    lr=1e-4,
    wd=None,
    epochs=50,
    d=256,
    d_hidden_factor=2.0,
    n_layers=4,
    hidden_dropout=0.0,
    residual_dropout=0.2,
    fit_cbs=[],
    seed=10,
    use_cuda=False,
):
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
            else ContTransformerRangeBoxCoxRange
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
        # FIXME: this is ugly, we probably should overload the metric setter and getter for the SurrogateEnsembleLearner
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
    if fit == "fit_flat_cos":
        surrogate.fit_flat_cos(epochs, lr=lr, wd=wd, cbs=fit_cbs)
    elif fit == "fit_one_cycle":
        surrogate.fit_one_cycle(epochs, lr_max=lr, wd=wd, cbs=fit_cbs)

    return surrogate


def tune_config_resnet(
    key, name, dl_train, use_cuda, tfms_fixed={}, trials=0, walltime=0, **kwargs
):
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
    # except for fit and wd
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
        fit = trial.suggest_categorical("fit", ["fit_flat_cos", "fit_one_cycle"])
        use_wd = trial.suggest_categorical("use_wd", [True, False])
        if use_wd:
            wd = trial.suggest_float("wd", 1e-7, 1e-2, log=True)
        else:
            wd = 0.0
        cbs = [FastAIPruningCallback(trial=trial, monitor="valid_loss")]

        surrogate = fit_config_resnet(
            key=key,
            dl_train=dl_train,
            tfms=tfms_fixed,
            fit=fit,
            lr=lr,
            wd=wd,
            d=d,
            d_hidden_factor=d_hidden_factor,
            n_layers=n_layers,
            hidden_dropout=hidden_dropout,
            residual_dropout=residual_dropout,
            fit_cbs=cbs,
            use_cuda=use_cuda,
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
                    partial(ContTransformerClamp, min=0.00, max=100.00),
                    ContTransformerRangeGrouped,
                ]
            ),
            "val_cross_entropy": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.00),
                    ContTransformerRangeGrouped,
                ]
            ),
            "val_balanced_accuracy": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.00, max=1.00),
                    ContTransformerRangeGrouped,
                ]
            ),
            "test_cross_entropy": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.00),
                    ContTransformerRangeGrouped,
                ]
            ),
            "test_balanced_accuracy": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.00, max=1.00),
                    ContTransformerRangeGrouped,
                ]
            ),
            "time": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.00),
                    ContTransformerRangeGrouped,
                ]
            ),
        }
    )
    tfms_list.update({"lcbench": tfms_lcbench})

    # tfms_nb301 = {}
    # tfms_list.update({"nb301": tfms_nb301})

    # tfms_rbv2_super = {}
    # tfms_list.update({"rbv2_super": tfms_rbv2_super})

    # tfms_rbv2_svm = {}
    # tfms_list.update({"rbv2_svm": tfms_rbv2_svm})

    # tfms_rbv2_xgboost = {}
    # tfms_list.update({"rbv2_xgboost": tfms_rbv2_xgboost})

    # tfms_rbv2_ranger = {}
    # tfms_list.update({"rbv2_ranger": tfms_rbv2_ranger})

    # tfms_rbv2_rpart = {}
    # tfms_list.update({"rbv2_rpart": tfms_rbv2_rpart})

    # tfms_rbv2_glmnet = {}
    # tfms_list.update({"rbv2_glmnet": tfms_rbv2_glmnet})

    # tfms_rbv2_aknn = {}
    # tfms_list.update({"rbv2_aknn": tfms_rbv2_aknn})

    tfms_iaml = {}
    tfms_iaml.update(
        {
            "mmce": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=1.0),
                    ContTransformerRangeGrouped,
                ]
            ),
            "f1": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=1.0),
                    ContTransformerRangeGrouped,
                ]
            ),
            "auc": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=1.0),
                    ContTransformerRangeGrouped,
                ]
            ),
            "logloss": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=None),
                    ContTransformerRangeGrouped,
                ]
            ),
            "rammodel": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=None),
                    ContTransformerRangeGrouped,
                ]
            ),
            "timetrain": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=None),
                    ContTransformerRangeGrouped,
                ]
            ),
            "mec": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=None),
                    ContTransformerRangeGrouped,
                ]
            ),
            "ias": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=None),
                    ContTransformerRangeGrouped,
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
                    ContTransformerRangeGrouped,
                ]
            ),
        }
    )
    tfms_list.update(
        {
            "iaml_glmnet": tfms_iaml,
            "iaml_rpart": tfms_iaml,
            "iaml_ranger": tfms_iaml,
            "iaml_xgboost": tfms_iaml,
            "iaml_super": tfms_iaml,
        }
    )

    tfms_fair = {}
    tfms_fair.update(
        {
            "mmce": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=1.0),
                    ContTransformerRangeGrouped,
                ]
            ),
            "f1": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=1.0),
                    ContTransformerRangeGrouped,
                ]
            ),
            "feo": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=1.0),
                    ContTransformerRangeGrouped,
                ]
            ),
            "fpredp": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=1.0),
                    ContTransformerRangeGrouped,
                ]
            ),
            "facc": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=1.0),
                    ContTransformerRangeGrouped,
                ]
            ),
            "ftpr": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=1.0),
                    ContTransformerRangeGrouped,
                ]
            ),
            "ffomr": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=1.0),
                    ContTransformerRangeGrouped,
                ]
            ),
            "ffnr": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=1.0),
                    ContTransformerRangeGrouped,
                ]
            ),
            "rammodel": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=None),
                    ContTransformerRangeGrouped,
                ]
            ),
            "timetrain": tfms_chain(
                [
                    partial(ContTransformerClamp, min=0.0, max=None),
                    ContTransformerRangeGrouped,
                ]
            ),
        }
    )
    tfms_list.update(
        {
            "fair_fgrrm": tfms_fair,
            "fair_rpart": tfms_fair,
            "fair_ranger": tfms_fair,
            "fair_xgboost": tfms_fair,
            "fair_super": tfms_fair,
        }
    )

    parser = argparse.ArgumentParser(description="Args for resnet tuning")
    parser.add_argument(
        "--key",
        type=str,
        default="iaml_glmnet",
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
        use_cuda=use_cuda,
        tfms_fixed=tfms_list.get(args.key),
        trials=args.trials,
        walltime=args.walltime,
    )

    best_params = study.best_params
    with open(Path(config.config_path, config.config.get("best_params")), "w") as f:
        json.dump(best_params, f)

    if not best_params.get("use_residual_dropout"):
        best_params.update({"residual_dropout": 0})

    if not best_params.get("use_wd"):
        best_params.update({"wd": 0})

    best_params.pop("use_residual_dropout")
    best_params.pop("use_wd")

    warnings.filterwarnings(
        "ignore", category=UserWarning
    )  # ignore warnings due to empty validation set
    surrogate = fit_config_resnet(
        args.key,
        dl_train=dl_refit,
        tfms=tfms_list.get(args.key),
        use_cuda=use_cuda,
        **best_params,
    )
    warnings.filterwarnings("default", category=UserWarning)  # reset warnings

    surrogate.export_onnx(config, device=device)
    generate_all_test_set_metrics(
        args.key, model=config.config.get("model"), save_to_csv=True
    )

    # FIXME: also fit noisy here
