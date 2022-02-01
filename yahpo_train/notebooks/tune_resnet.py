from yahpo_train.models import *
from yahpo_train.learner import *
from yahpo_train.metrics import *
from yahpo_train.cont_scalers import *
from yahpo_gym import benchmark_set
from yahpo_gym.benchmarks import lcbench, rbv2, nasbench_301, fcnet, taskset, iaml
from yahpo_gym.configuration import cfg
from fastai.callback.wandb import *
from functools import partial
import wandb
import argparse

def fit_config_resnet(key, dls_train=None, save_df_test_encoding=True, embds_dbl=None, embds_tgt=None, tfms=None, lr=1e-4, epochs=100, d=256, d_hidden_factor=2., n_layers=4, hidden_dropout=0., residual_dropout=.2, bs=10240, frac=1., mixup=True, export=False, log_wandb=True, wandb_entity='mfsurrogates', cbs=[], device='cuda:0'):
    """
    Fit function with hyperparameters for resnet.
    """
    cc = cfg(key)

    if dls_train is None:
        dls_train = dl_from_config(cc, bs=bs, frac=frac, save_df_test=save_df_test_encoding, save_encoding=save_df_test_encoding)  # train_frac is set to 0.8, and valid frac within train frac to 0.2, so for frac = 1 we have 0.6 train, 0.2 valid and 0.2 test

    # Construct embds from transforms
    # tfms overwrites emdbs_dbl, embds_tgt
    if tfms is not None:
        embds_dbl = [tfms.get(name) if tfms.get(name) is not None else ContTransformerRange for name, cont in dls_train.all_cols[dls_train.cont_names].iteritems()]
        embds_tgt = [tfms.get(name) if tfms.get(name) is not None else ContTransformerRange for name, cont in dls_train.ys.iteritems()]

    # Instantiate learner
    f = ResNet(dls_train, embds_dbl=embds_dbl, embds_tgt=embds_tgt, d=d, d_hidden_factor=d_hidden_factor, n_layers=n_layers, hidden_dropout=hidden_dropout, residual_dropout=residual_dropout)
    l = SurrogateTabularLearner(dls_train, f, loss_func=nn.MSELoss(reduction='mean'), metrics=nn.MSELoss)
    l.metrics = [AvgTfedMetric(mae), AvgTfedMetric(r2), AvgTfedMetric(spearman), AvgTfedMetric(napct)]
    if mixup:
        l.add_cb(MixHandler)
    l.add_cb(EarlyStoppingCallback(patience=10))
    if len(cbs):
        [l.add_cb(cb) for cb in cbs]

    # Log results to wandb
    if log_wandb:
        wandb.init(project=key, entity=wandb_entity)
        l.add_cb(WandbMetricsTableCallback())
        wandb.config.update({'cont_tf': l.embds_dbl, 'tgt_tf': l.embds_tgt, 'fraction': frac,}, allow_val_change=True)
        wandb.config.update({'deep': deep, 'deeper': deeper, 'dropout':dropout, 'wide':wide, 'use_bn':use_bn}, allow_val_change=True)

    # Fit
    l.fit_flat_cos(epochs, lr)

    if log_wandb: 
        wandb.finish()

    if export:
        l.export_onnx(cc, device=device)

    return l


def tune_config_resnet(key, name, tfms_fixed={}, trials=1000, walltime=86400, **kwargs):
    import optuna
    from optuna.integration import FastAIPruningCallback
    from optuna.visualization import plot_optimization_history
    import logging
    import sys

    if trials == 0:
        trials = None

    if walltime == 0:
        walltime = None

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    
    storage_name = "sqlite:///{}.db".format(name)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(study_name=name, storage=storage_name, direction="minimize", pruner=pruner, load_if_exists=True)

    cc = cfg(key)
    dls_train = dl_from_config(cc, bs=10240, frac=1.0)  # set data loader up once, also save dls_train and encoding

    trange = ContTransformerRange
    tlog = ContTransformerLogRange
    tlog2 = ContTransformerLog2Range
    tnexp = ContTransformerNegExpRange
    tclamp = ContTransformerClamp01Range
    trafos = {"trange":trange, "tlog":tlog, "tlog2":tlog2, "tnexp":tnexp, "tclamp":tclamp}

    # for the search space see https://arxiv.org/pdf/2106.11959.pdf

    def objective(trial):
        tfms = copy(tfms_fixed)
        for y in cc.y_names:
            if y not in tfms.keys():  # exclude variables provided in tfms_fixed
                # if opt_tfms_y is False use ContTransformerRange
                opt_tfms_y = trial.suggest_categorical("opt_tfms_" + y, [True, False])
                if opt_tfms_y:
                    tf = trial.suggest_categorical("tfms_" + y, ["tlog", "tnexp", "tclamp"])
                else:
                    tf = "trange"
                tfms.update({y:trafos.get(tf)})
        for x in cc.cont_names:
            if x not in tfms.keys():  # exclude variables provided in tfms_fixed
                # if opt_tfms_x is False use ContTransformerRange
                opt_tfms_x = trial.suggest_categorical("opt_tfms_" + x, [True, False])
                if opt_tfms_x:
                    tf = trial.suggest_categorical("tfms_" + x, ["tlog", "tlog2", "tnexp"])
                else:
                    tf = "trange"
                tfms.update({x:trafos.get(tf)})

        d = trial.suggest_int("d", 64, 1024, step = 64)  # layer size
        d_hidden_factor = trial.suggest_float("d_hidden_factor", 1., 4.)  # hidden factor
        n_layers = trial.suggest_int("n_layers", 1, 8)  # number of layers
        hidden_dropout = trial.suggest_float("hidden_dropout", 0., 0.5)  # hidden dropout
        use_residual_dropout = trial.suggest_categorical("use_residual_dropout", [True, False])
        if use_residual_dropout:
            residual_dropout = trial.suggest_float("residual_dropout", 1e-2, 0.5)
        else:
            residual_dropout = 0.
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        mixup = trial.suggest_categorical("mixup", [True, False])
        cbs = [FastAIPruningCallback(trial=trial, monitor='valid_loss')]
        
        l = fit_config_resnet(key=key, dls_train=dls_train, tfms=tfms, lr=lr, d=d, d_hidden_factor=d_hidden_factor, n_layers=n_layers, hidden_dropout=hidden_dropout, residual_dropout=residual_dropout, mixup=mixup, log_wandb=False, cbs=cbs, **kwargs)
        loss = l.recorder.final_record.items[1]  # [1] is validation loss
        return loss
    
    study.optimize(objective, n_trials=trials, timeout=walltime)
    # plot_optimization_history(study)
    return study


def fit_from_best_params_resnet(key, best_params, tfms_fixed={}, log_wandb=False, **kwargs):
    cc = cfg(key)
    tfms = copy(tfms_fixed)

    trange = ContTransformerRange
    tlog = ContTransformerLogRange
    tlog2 = ContTransformerLog2Range
    tnexp = ContTransformerNegExpRange
    tclamp = ContTransformerClamp01Range
    trafos = {"trange":trange, "tlog":tlog, "tlog2":tlog2, "tnexp":tnexp, "tclamp":tclamp}

    for y in cc.y_names:
        if y not in tfms.keys():  # exclude variables provided in tfms_fixed
            # if opt_tfms_y is False use ContTransformerRange
            opt_tfms_y = best_params.get("opt_tfms_" + y)
            if opt_tfms_y:
                tf = best_params.get("tfms_" + y)
            else:
                tf = "trange"
            tfms.update({y:trafos.get(tf)})
    for x in cc.cont_names:
        if x not in tfms.keys():  # exclude variables provided in tfms_fixed
            # if opt_tfms_x is False use ContTransformerRange
            opt_tfms_x = best_params.get("opt_tfms_" + x)
            if opt_tfms_x:
                tf = best_params.get("tfms_" + x)
            else:
                tf = "trange"
            tfms.update({x:trafos.get(tf)})

    d = best_params.get("d")
    d_hidden_factor = best_params.get("d_hidden_factor")
    n_layers = best_params.get("n_layers")
    hidden_dropout = best_params.get("hidden_dropout")
    if best_params.get("use_residual_dropout"):
        residual_dropout = best_params.get("residual_dropout")
    else:
        residual_dropout = 0.
    lr = best_params.get("lr")
    mixup = best_params.get("mixup")
    
    l = fit_config_resnet(key=key, tfms=tfms, lr=lr, d=d, d_hidden_factor=d_hidden_factor, n_layers=n_layers, hidden_dropout=hidden_dropout, residual_dropout=residual_dropout, mixup=mixup, log_wandb=log_wandb, **kwargs)

    return l

if __name__ == '__main__':

    # tfms_list holds for each benchmark scenario (key) optional transformers that should be fixed and not tuned

    tfms_list = {}

    tfms_lcbench = {}  # FIXME:
    tfms_list.update({"lcbench":tfms_lcbench})

    tfms_nb301 = {}  # FIXME:
    tfms_list.update({"nb301":tfms_nb301})

    tfms_rbv2_super = {}  # FIXME:
    tfms_list.update({"rbv2_super":tfms_rbv2_super})

    tfms_rbv2_svm = {}  # FIXME:
    tfms_list.update({"rbv2_svm":tfms_rbv2_svm})

    tfms_rbv2_xgboost = {}  # FIXME:
    tfms_list.update({"rbv2_xgboost":tfms_rbv2_xgboost})

    tfms_rbv2_ranger = {}  # FIXME:
    tfms_list.update({"rbv2_ranger":tfms_rbv2_ranger})

    tfms_rbv2_rpart = {}  # FIXME:
    tfms_list.update({"rbv2_rpart":tfms_rbv2_rpart})

    tfms_rbv2_glmnet = {}  # FIXME:
    tfms_list.update({"rbv2_glmnet":tfms_rbv2_glmnet})

    tfms_rbv2_aknn = {}  # FIXME:
    tfms_list.update({"rbv2_aknn":tfms_rbv2_aknn})

    tfms_iaml_super = {}
    [tfms_iaml_super.update({k:tfms_chain([ContTransformerInt, ContTransformerRange])}) for k in ["nf"]]
    tfms_list.update({"iaml_super":tfms_iaml_super})

    tfms_iaml_xgboost = {}
    [tfms_iaml_xgboost.update({k:tfms_chain([ContTransformerInt, ContTransformerRange])}) for k in ["nf"]]
    tfms_list.update({"iaml_xgboost":tfms_iaml_xgboost})

    tfms_iaml_ranger = {}
    [tfms_iaml_ranger.update({k:tfms_chain([ContTransformerInt, ContTransformerRange])}) for k in ["nf"]]
    tfms_list.update({"iaml_ranger":tfms_iaml_ranger})

    tfms_iaml_rpart = {}
    [tfms_iaml_rpart.update({k:tfms_chain([ContTransformerInt, ContTransformerRange])}) for k in ["nf"]]
    tfms_list.update({"iaml_rpart":tfms_iaml_rpart})

    tfms_iaml_glmnet = {}
    [tfms_iaml_glmnet.update({k:tfms_chain([ContTransformerInt, ContTransformerRange])}) for k in ["nf"]]
    tfms_list.update({"iaml_glmnet":tfms_iaml_glmnet})

    # FIXME: fcnet, taskset

    parser = argparse.ArgumentParser(description='Args for resnet tuning')
    parser.add_argument('--key', type=str, default="iaml_glmnet", help='Key of benchmark scenario, e.g., "iaml_glmnet"')    
    parser.add_argument('--name', type=str, default="tune_iaml_glmnet_resnet", help='Name of the optuna study, e.g., "tune_iaml_glmnet_resnet"')
    parser.add_argument('--trials', type=int, default=0, help='Number of optuna trials')  # by default we run until terminated externally
    parser.add_argument('--walltime', type=int, default=0, help='Walltime for optuna timeout in seconds') # by default we run until terminated externally
    args = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name()
        print("Using cuda device: " + device_name + " " + str(current_device))
    else:
        raise ValueError("No cuda device available. You probably do not want to tune on CPUs.")

    tune_config_resnet(args.key, name=args.name, tfms_fixed=tfms_list.get(args.key), trials=args.trials, walltime=args.walltime)

#if __name__ == '__main__':
#    wandb.login()
#    #device = torch.device("cpu")
#    device = 'cuda:0'
#    # general tuning example workflow
#    # 1. specify transformers you want to consider fixed (and not tuned over)
#    tfms_xgboost = {}
#    tfms_xgboost.update({"nf":tfms_chain([ContTransformerInt, ContTransformerRange])})
#    # 2. tune by providing the fixed transformers (if any)
#    study_xgboost = tune_config_resnet("iaml_xgboost", name="tune_iaml_xgboost_new", tfms_fixed=tfms_xgboost)
#    # 3. extract the best params and refit the model (FIXME: could refit on whole train + valid data?)
#    #    set export = True so that the onnx model is exported
#    #    caveat: this overwrites the exiting model! # FIXME: should versionize this automatically
#    l = fit_from_best_params_resnet("iaml_xgboost", best_params=study_xgboost.best_params, tfms_fixed=tfms_xgboost, export=True, device=device, epochs=100)
#    # 4. get the performance metrics on the test set relying on the newly exported onnx model
#    get_testset_metrics("iaml_xgboost")
#
#    # load existing one:
#    name = storage_name = "sqlite:///{}.db".format("tune_iaml_xgboost_new")
#    study_xgboost = optuna.load_study(None, storage = name)
#
#    # tfms_super nf tfms_chain([ContTransformerInt, ContTransformerRange]
#    tfms_super.update({"timetrain":ContTransformerLogRange})
#    tfms_super.update({"timepredict":ContTransformerLogRange})
#    tfms_super.update({"rammodel":ContTransformerLogRange})
#    tfms_super.update({"ias":ContTransformerLogRange})
#    fit_from_best_params_resnet("iaml_super", study_super.best_params, tfms_fixed=tfms_super, log_wandb=True, export=True, epochs=100)

