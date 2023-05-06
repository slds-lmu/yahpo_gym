from yahpo_train.models import *
from yahpo_train.models_ensemble import *
from yahpo_train.learner import *
from yahpo_train.losses import *
from yahpo_train.metrics import *
from yahpo_train.cont_scalers import *
from yahpo_gym import benchmark_set
from yahpo_gym.benchmarks import lcbench, rbv2, nb301, iaml, fair
from yahpo_gym.configuration import cfg
from functools import partial
import argparse
import optuna
from optuna.integration import FastAIPruningCallback
from optuna.visualization import plot_optimization_history
import torch.nn as nn
import logging
import warnings


def fit_config_resnet(
    key,
    dls_train,
    noisy=False,
    embds_dbl=None,
    embds_tgt=None,
    tfms=None,
    fit="fit_flat_cos",
    lr=1e-4,
    wd=None,
    epochs=50,  # FIXME:
    d=256,
    d_hidden_factor=2.0,
    n_layers=4,
    hidden_dropout=0.0,
    residual_dropout=0.2,
    fit_cbs=[],
    export=False,
    export_device="cuda:0",
):
    """
    Fit function with hyperparameters for resnet.
    """
    cc = cfg(key)

    # Construct embds from tfms
    # tfms overwrites emdbs_dbl, embds_tgt
    if tfms is not None:
        embds_dbl = [
            tfms.get(name) if tfms.get(name) is not None else ContTransformerStandardize
            for name, cont in dls_train.all_cols[dls_train.cont_names].items()
        ]
        embds_tgt = [
            tfms.get(name)
            if tfms.get(name) is not None
            else (
                ContTransformerStandardizeGroupedRange
                if cc.instance_names is not None
                else ContTransformerStandardizeRange
            )
            for name, cont in dls_train.ys.items()
        ]

    # Instantiate learner
    if noisy:
        model = Ensemble(
            ResNet,
            n_models=3,
            dls=dls_train,
            embds_dbl=embds_dbl,
            embds_tgt=embds_tgt,
            instance_names=cc.instance_names,
            d=d,
            d_hidden_factor=d_hidden_factor,
            n_layers=n_layers,
            hidden_dropout=hidden_dropout,
            residual_dropout=residual_dropout,
        )
        surrogate = SurrogateEnsembleLearner(
            dls_train, ensemble=model, loss_func=MultiMseLoss()
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
            dls_train,
            embds_dbl=embds_dbl,
            embds_tgt=embds_tgt,
            instance_names=cc.instance_names,
            d=d,
            d_hidden_factor=d_hidden_factor,
            n_layers=n_layers,
            hidden_dropout=hidden_dropout,
            residual_dropout=residual_dropout,
        )
        surrogate = SurrogateTabularLearner(
            dls_train, model=model, loss_func=MultiMseLoss()
        )
        surrogate.metrics = [
            AvgTfedMetric(mae),
            AvgTfedMetric(r2),
            AvgTfedMetric(spearman),
            AvgTfedMetric(pearson),
            AvgTfedMetric(napct),
        ]

    # Fit
    cbs = [
        # EarlyStoppingCallback(patience=100),
        SaveModelCallback(
            monitor="valid_loss",
            comp=np.less,
            fname="best",
            with_opt=True,
            reset_on_fit=True,
        ),
    ]
    cbs += fit_cbs
    if fit == "fit_flat_cos":
        surrogate.fit_flat_cos(epochs, lr=lr, wd=wd, cbs=cbs)
    elif fit == "fit_one_cycle":
        surrogate.fit_one_cycle(epochs, lr_max=lr, wd=wd, cbs=cbs)

    surrogate = surrogate.load("best")

    if export:
        surrogate.export_onnx(cc, device=export_device)

    return surrogate


def tune_config_resnet(
    key, name, device, tfms_fixed={}, trials=1000, walltime=86400, **kwargs
):
    if trials == 0:
        trials = None

    if walltime == 0:
        walltime = None

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    storage_name = "sqlite:///{}.db".format(name)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(
        study_name=name,
        storage=storage_name,
        direction="minimize",
        pruner=pruner,
        load_if_exists=True,
    )

    cc = cfg(key)
    # dls_train = dl_from_config(
    #    cc, pin_memory=True, device=device, save_df_test=True, save_encoding=True
    # )
    dls_train = dl_from_config(cc, save_df_test=True, save_encoding=True)

    # for the search space see https://arxiv.org/pdf/2106.11959.pdf
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
            dls_train=dls_train,
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
            **kwargs
        )
        loss = surrogate.recorder.final_record.items[1]  # [1] is validation loss
        return loss

    study.optimize(objective, n_trials=trials, timeout=walltime)
    # plot_optimization_history(study)
    return study


def fit_from_best_params_resnet(key, best_params, noisy=False, tfms_fixed={}, **kwargs):
    d = best_params.get("d")
    d_hidden_factor = best_params.get("d_hidden_factor")
    n_layers = best_params.get("n_layers")
    hidden_dropout = best_params.get("hidden_dropout")
    if best_params.get("use_residual_dropout"):
        residual_dropout = best_params.get("residual_dropout")
    else:
        residual_dropout = 0.0
    lr = best_params.get("lr")

    # FIXME: where to get the dls_train from?

    surrogate = fit_config_resnet(
        key=key,
        tfms=tfms_fixed,
        noisy=noisy,
        lr=lr,
        d=d,
        d_hidden_factor=d_hidden_factor,
        n_layers=n_layers,
        hidden_dropout=hidden_dropout,
        residual_dropout=residual_dropout,
        **kwargs
    )

    return surrogate


if __name__ == "__main__":
    # tfms_list holds for each benchmark scenario (key) optional transformers that should be fixed and not tuned
    tfms_list = {}

    # tfms_lcbench = {}
    # tfms_list.update({"lcbench": tfms_lcbench})

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

    # tfms_iaml_super = {}
    # tfms_iaml_super.update({"nf": tfms_chain([ContTransformerInt, ContTransformerStandardize])})
    # tfms_list.update({"iaml_super": tfms_iaml_super})

    # tfms_iaml_xgboost = {}
    # tfms_iaml_xgboost.update({"nf": tfms_chain([ContTransformerInt, ContTransformerStandardize])})
    # tfms_list.update({"iaml_xgboost": tfms_iaml_xgboost})

    # tfms_iaml_ranger = {}
    # tfms_iaml_ranger.update({"nf": tfms_chain([ContTransformerInt, ContTransformerStandardize])})
    # tfms_list.update({"iaml_ranger": tfms_iaml_ranger})

    # tfms_iaml_rpart = {}
    # tfms_iaml_rpart.update({"nf": tfms_chain([ContTransformerInt, ContTransformerStandardize])})
    # tfms_list.update({"iaml_rpart": tfms_iaml_rpart})

    tfms_iaml_glmnet = {}
    tfms_iaml_glmnet.update(
        {"nf": tfms_chain([ContTransformerInt, ContTransformerStandardize])}
    )
    tfms_list.update({"iaml_glmnet": tfms_iaml_glmnet})

    # tfms_fair_fgrrm = {}
    # tfms_list.update({"fair_fgrrm": tfms_fair_fgrrm})

    # tfms_fair_rpart = {}
    # tfms_list.update({"fair_rpart": tfms_fair_rpart})

    # tfms_fair_ranger = {}
    # tfms_list.update({"fair_ranger": tfms_fair_ranger})

    # tfms_fair_xgboost = {}
    # tfms_list.update({"fair_xgboost": tfms_fair_xgboost})

    # tfms_fair_super = {}
    # tfms_list.update({"fair_super": tfms_fair_super})

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
        print("Using cuda device: " + device_name + " " + str(current_device))
    else:
        warnings.warn(
            "No cuda device available. You probably do not want to tune on CPUs."
        )
        device = torch.device("cpu")

    tune_config_resnet(
        args.key,
        name=args.name,
        device=device,
        tfms_fixed=tfms_list.get(args.key),
        trials=args.trials,
        walltime=args.walltime,
    )
