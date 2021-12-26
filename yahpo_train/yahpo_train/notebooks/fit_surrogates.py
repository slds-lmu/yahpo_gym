from yahpo_train.model import *
from yahpo_train.metrics import *
from yahpo_train.cont_scalers import *
from yahpo_gym import benchmark_set
from yahpo_gym.benchmarks import lcbench, rbv2, nasbench_301, fcnet, taskset, iaml
from yahpo_gym.configuration import cfg
from fastai.callback.wandb import *
from functools import partial
import wandb

def fit_config(key, dls_train=None, save_df_test_encoding=True, embds_dbl=None, embds_tgt=None, tfms=None, lr=1e-4, epochs=100, deep=[1024,512,256], deeper=[], dropout=0., wide=True, use_bn=False, bs=10240, frac=1., mixup=True, export=False, log_wandb=True, wandb_entity='mfsurrogates', cbs=[], device='cuda:0'):
    """
    Fit function with hyperparameters.
    """
    cc = cfg(key)

    if dls_train is None:
        dls_train = dl_from_config(cc, bs=bs, frac=frac, save_df_test=save_df_test_encoding, save_encoding=save_df_test_encoding)  # train_frac is set to 0.9, so for frac = 1 we have 0.72 train, 0.18 valid and 0.1 test

    # Construct embds from transforms
    # tfms overwrites emdbs_dbl, embds_tgt
    if tfms is not None:
        embds_dbl = [tfms.get(name) if tfms.get(name) is not None else ContTransformerRange for name, cont in dls_train.all_cols[dls_train.cont_names].iteritems()]
        embds_tgt = [tfms.get(name) if tfms.get(name) is not None else ContTransformerRange for name, cont in dls_train.ys.iteritems()]

    # Instantiate learner
    f = FFSurrogateModel(dls_train, layers=deep, deeper=deeper, ps=dropout, use_bn=use_bn, wide=wide, embds_dbl=embds_dbl, embds_tgt=embds_tgt)
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


def get_testset_metrics(key):
    bench = benchmark_set.BenchmarkSet(key)
    bench.check = False  # see note below
    dtypes = dict(zip(bench.config.cat_names, ["object"] * len(bench.config.cat_names)))
    dtypes.update(dict(zip(bench.config.cont_names+bench.config.y_names, ["float32"] * len(bench.config.cont_names+bench.config.y_names))))
    df = pd.read_csv(bench.config.get_path("test_dataset"), dtype=dtypes)

    x = df[bench.config.hp_names]
    truth = df[bench.config.y_names]
    # note that the following is somewhat unsafe: we assume that dtypes are correctly represented as expected by the ConfigSpace
    response = x.apply(lambda point: bench.objective_function(point[~point.isna()].to_dict()), axis=1, result_type="expand")
    truth_tensor = torch.tensor(truth.values)
    response_tensor = torch.tensor(response.values)

    metrics_dict = {}
    metrics = {"mae":mae, "r2":r2, "spearman":spearman}
    for metric_name,metric in zip(metrics.keys(), metrics.values()):
        values = metric(truth_tensor, response_tensor)
        metrics_dict.update({metric_name:dict(zip([y + "_" + metric_name for y in bench.config.y_names], [*values]))})

    return metrics_dict


def get_arch(max_units, n, shape):
    if max_units == 0:
        return []
    if n == 0:
       n = 4
    if shape == "square":
        return [2**max_units for x in range(n)]
    if shape == "cone":
        units = [2**max_units]
        for x in range(n):
            units += [int(units[-1]/2)]
        return units


def tune_config(key, name, tfms_fixed={}, **kwargs):
    import optuna
    from optuna.integration import FastAIPruningCallback
    from optuna.visualization import plot_optimization_history
    import logging
    import sys

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
    trafos = {"trange":trange, "tlog":tlog, "tlog2":tlog2, "tnexp":tnexp, "tclamp":tclamp}

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

        opt_deep_arch = trial.suggest_categorical("opt_deep_arch", [True, False])
        if opt_deep_arch:
            deep_u = trial.suggest_categorical("deep_u", [7, 8, 9, 10])
            deep_n = trial.suggest_categorical("deep_n", [0, 1, 2, 3])
            deep_s = trial.suggest_categorical("deep_s", ["square", "cone"])
            deep = get_arch(deep_u, deep_n, deep_s)
            use_deeper = trial.suggest_categorical("use_deeper", [True, False])
            if use_deeper:
                deeper_u = trial.suggest_categorical("deeper_u", [7, 8, 9, 10])
                deeper = get_arch(deeper_u, deep_n + 2, deep_s)
            else:
                deeper = []
        else:
            deep = [1024,512,256]
            deeper = []

        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        wide = trial.suggest_categorical("wide", [True, False])
        mixup = trial.suggest_categorical("mixup", [True, False])
        use_bn = trial.suggest_categorical("use_bn", [True, False])
        dropout = trial.suggest_categorical("dropout", [0., 0.25, 0.5])
        cbs = [FastAIPruningCallback(trial=trial, monitor='valid_loss')]
        
        l = fit_config(key=key, dls_train=dls_train, tfms=tfms, lr=lr, deep=deep, deeper=deeper, wide=wide, mixup=mixup, use_bn=use_bn, dropout=dropout, log_wandb=False, cbs=cbs, **kwargs)
        return l.recorder.losses[-1]
    
    study.optimize(objective, n_trials=1000, timeout=86400)
    # plot_optimization_history(study)
    return study


def fit_from_best_params(key, best_params, tfms_fixed={}, log_wandb=False, **kwargs):
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

    if best_params.get("opt_deep_arch"):
        deep = get_arch(best_params.get("deep_u"), best_params.get("deep_n"), best_params.get("deep_s"))
        use_deeper = best_params.get("use_deeper")
        if use_deeper:
            deeper = get_arch(best_params.get("deeper_u"), best_params.get("deep_n") + 2, best_params.get("deep_s"))
        else:
            deeper = []
    else:
        deep = [1024,512,256]
        deeper = []

    lr = best_params.get("lr")
    wide = best_params.get("wide")
    mixup = best_params.get("mixup")
    use_bn = best_params.get("use_bn")
    dropout = best_params.get("dropout")
    
    l = fit_config(key=key, tfms=tfms, lr=lr, deep=deep, deeper=deeper, wide=wide, mixup=mixup, use_bn=use_bn, dropout=dropout, log_wandb=log_wandb, **kwargs)
    return l


def fit_nb301(key='nb301', **kwargs):
    embds_dbl = [partial(ContTransformerMultScalar, m=1/52)]
    embds_tgt = [partial(ContTransformerMultScalar, m=1/100), ContTransformerRange]
    fit_config(key, embds_dbl=embds_dbl, embds_tgt=embds_tgt, **kwargs)
        
def fit_rbv2_super(key='rbv2_super', **kwargs):
    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerClamp01Range}) for k in ["mmce", "f1", "auc"]]
    [tfms.update({k:ContTransformerRange}) for k in ["aknn.k", "aknn.M", "rpart.maxdepth", "rpart.minsplit", "rpart.minbucket", "xgboost.max_depth"]]
    [tfms.update({k:tfms_chain([ContTransformerLog, ContTransformerRange])}) for k in ["svm.cost", "svm.gamma"]]
    [tfms.update({k:ContTransformerClamp0LogRange}) for k in ["timetrain", "timepredict"]]
    [tfms.update({k:tfms_chain([partial(ContTransformerLog, logfun=torch.log2, expfun=torch.exp2), ContTransformerRange])}) for k in ["glmnet.s", "rpart.cp", "aknn.ef", "aknn.ef_construction", "xgboost.nrounds", "xgboost.eta", "xgboost.gamma", "xgboost.lambda", "xgboost.alpha", "xgboost.min_child_weight", "ranger.num.trees", "ranger.min.node.size", 'ranger.num.random.splits']]
    [tfms.update({k:tfms_chain([partial(ContTransformerNegExp), ContTransformerRange])}) for k in ["logloss"]]
    fit_config(key, tfms=tfms, **kwargs)
  
def fit_rbv2_svm(key='rbv2_svm', **kwargs):
    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerClamp01Range}) for k in ["mmce", "f1", "auc"]]
    [tfms.update({k:ContTransformerClamp0LogRange}) for k in ["timetrain", "timepredict"]]
    [tfms.update({k:tfms_chain([partial(ContTransformerLog, logfun=torch.log2, expfun=torch.exp2), ContTransformerRange])}) for k in ["cost", "gamma"]]
    [tfms.update({k:tfms_chain([partial(ContTransformerNegExp), ContTransformerRange])}) for k in ["logloss"]]
    fit_config(key, tfms=tfms, **kwargs)

def fit_rbv2_xgboost(key='rbv2_xgboost', **kwargs):
    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerClamp01Range}) for k in ["mmce", "f1", "auc"]]
    [tfms.update({k:ContTransformerRange}) for k in ["max_depth"]]
    [tfms.update({k:ContTransformerClamp0LogRange}) for k in ["timetrain", "timepredict"]]
    [tfms.update({k:tfms_chain([partial(ContTransformerLog, logfun=torch.log2, expfun=torch.exp2), ContTransformerRange])})for k in ["nrounds", "eta", "gamma", "lambda", "alpha", "min_child_weight"]]
    [tfms.update({k:tfms_chain([partial(ContTransformerNegExp), ContTransformerRange])}) for k in ["logloss"]]
    fit_config(key, tfms=tfms, **kwargs)

def fit_rbv2_ranger(key='rbv2_ranger', **kwargs):
    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerClamp01Range}) for k in ["mmce", "f1", "auc"]]
    [tfms.update({k:ContTransformerClamp0LogRange}) for k in ["timetrain", "timepredict"]]
    [tfms.update({k:ContTransformerLog2Range}) for k in ["num.trees", "min.node.size", 'num.random.splits']]
    [tfms.update({k:tfms_chain([partial(ContTransformerNegExp), ContTransformerRange])}) for k in ["logloss"]]
    fit_config(key, tfms=tfms, **kwargs)

def fit_rbv2_rpart(key='rbv2_rpart', **kwargs):
    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerClamp01Range}) for k in ["mmce", "f1", "auc"]]
    [tfms.update({k:ContTransformerRange}) for k in ["maxdepth", "minsplit", "minbucket"]]
    [tfms.update({k:ContTransformerClamp0LogRange}) for k in ["timetrain", "timepredict"]]
    [tfms.update({k:ContTransformerLog2Range}) for k in ["cp"]]
    [tfms.update({k:tfms_chain([partial(ContTransformerNegExp), ContTransformerRange])}) for k in ["logloss"]]
    fit_config(key, tfms=tfms, **kwargs)
    
def fit_rbv2_glmnet(key='rbv2_glmnet', **kwargs):
    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerClamp01Range}) for k in ["mmce", "f1", "auc"]]
    [tfms.update({k:ContTransformerClamp0LogRange}) for k in ["timetrain", "timepredict"]]
    [tfms.update({k:ContTransformerLog2Range}) for k in ["s"]]
    [tfms.update({k:ContTransformerRange}) for k in ["repl"]]
    [tfms.update({k:tfms_chain([partial(ContTransformerNegExp), ContTransformerRange])}) for k in ["logloss"]]
    fit_config(key, tfms=tfms, **kwargs)

def fit_rbv2_aknn(key='rbv2_aknn', **kwargs):
    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerClamp01Range}) for k in ["mmce", "f1", "auc"]]
    [tfms.update({k:ContTransformerRange}) for k in ["k", "M"]]
    [tfms.update({k:ContTransformerClamp0LogRange}) for k in ["timetrain", "timepredict"]]
    [tfms.update({k:ContTransformerLog2Range}) for k in ["ef", "ef_construction"]]
    [tfms.update({k:tfms_chain([partial(ContTransformerNegExp), ContTransformerRange])}) for k in ["logloss"]]
    fit_config(key, tfms=tfms, **kwargs)

def fit_fcnet(key='fcnet', **kwargs):
    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ["batch_size", "n_units_1", "n_units_2"]]
    [tfms.update({k:ContTransformerLog2Range}) for k in ["init_lr", "runtime", "n_params"]]
    [tfms.update({k:tfms_chain([partial(ContTransformerNegExp), ContTransformerRange])}) for k in ["valid_loss"]]
    fit_config(key, tfms=tfms, **kwargs)

def fit_lcbench(key='lcbench', **kwargs):
    # Transforms
    tfms = {}
    [tfms.update({k:tfms_chain([partial(ContTransformerClamp, min=0., max=1.), ContTransformerRange])}) for k in ["val_accuracy", "val_balanced_accuracy", "test_balanced_accuracy"]]
    [tfms.update({k:ContTransformerRange}) for k in ["batch_size", "max_units"]]
    [tfms.update({k:ContTransformerLog2Range}) for k in ["time"]]
    [tfms.update({k:tfms_chain([partial(ContTransformerNegExp), ContTransformerRange])}) for k in  ["val_cross_entropy", "test_cross_entropy"]]
    fit_config(key, tfms=tfms, **kwargs)

def fit_taskset(key='taskset', **kwargs):
    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ['replication']]
    [tfms.update({k:tfms_chain([partial(ContTransformerLog, logfun=torch.log2, expfun=torch.exp2), ContTransformerRange])}) for k in ["epoch"]]
    [tfms.update({k:tfms_chain([partial(ContTransformerLog, logfun=torch.log, expfun=torch.exp), ContTransformerRange])})  for k in ["learning_rate", 'beta1', 'beta2', 'epsilon', 'l1', 'l2', 'linear_decay', 'exponential_decay']]
    [tfms.update({k:tfms_chain([partial(ContTransformerNegExp), ContTransformerRange])}) for k in  ["train", "valid1", "valid2", "test"]]
    fit_config(key, tfms=tfms, **kwargs)

def fit_iaml_ranger(key='iaml_ranger', **kwargs):
    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerInt}) for k in ["nf"]]
    [tfms.update({k:ContTransformerRange}) for k in ["mmce", "f1", "auc", "mec"]]
    [tfms.update({k:ContTransformerLogRange}) for k in ["timetrain", "timepredict", "ramtrain", "rammodel", "rampredict", "ias"]]
    [tfms.update({k:ContTransformerLog2Range}) for k in ["num.trees", "min.node.size", 'num.random.splits']]
    [tfms.update({k:partial(ContTransformerNegExpRange, q=.975)}) for k in ["logloss"]]
    return fit_config(key, tfms=tfms, **kwargs)

def fit_iaml_rpart(key='iaml_rpart', **kwargs):
    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ["mmce", "f1", "auc", "mec", "ias", "nf", "maxdepth", "minsplit", "minbucket"]]
    [tfms.update({k:ContTransformerLogRange}) for k in ["timetrain", "timepredict", "ramtrain", "rammodel", "rampredict"]]
    [tfms.update({k:ContTransformerLog2Range}) for k in ["cp"]]
    [tfms.update({k:ContTransformerNegExpRange}) for k in ["logloss"]]
    return fit_config(key, tfms=tfms, **kwargs)

def fit_iaml_glmnet(key='iaml_glmnet', **kwargs):
    # Transforms
    tfms = {}
    [tfms.update({k:tfms_chain([ContTransformerInt, ContTransformerRange])}) for k key, best_params, tfms_fixed={}, log_wandb=False, **kwargs):
in ["nf"]]
    [tfms.update({k:partial(ContTransformerRange)}) for k in ["auc", "ias", "mec", "mmce", "rammodel", "ramtrain", "timepredict"]]
    [tfms.update({k:partial(ContTransformerLogRange)}) for k in ["alpha", "rampredict", "timetrain", "logloss", "s", "trainsize"]]
    [tfms.update({k:partial(ContTransformerNegExpRange)}) for k in ["f1"]]
    return fit_config(key, tfms=tfms, **kwargs)

def fit_iaml_xgboost(key='iaml_xgboost', **kwargs):
    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ["mmce", "f1", "auc", "mec", "ias", "nf", "max_depth"]]
    [tfms.update({k:partial(ContTransformerLogRange)}) for k in ["timetrain", "timepredict", "ramtrain", "rammodel", "rampredict"]]
    [tfms.update({k:ContTransformerLog2Range}) for k in ["nrounds", "eta", "gamma", "lambda", "alpha", "min_child_weight"]]
    [tfms.update({k:ContTransformerNegExpRange}) for k in ["logloss"]]
    return fit_config(key, tfms=tfms, **kwargs)

def fit_iaml_super(key='iaml_super', **kwargs):
    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ["mmce", "f1", "auc", "mec", "nf", "rpart.maxdepth", "rpart.minsplit", "rpart.minbucket", "xgboost.max_depth"]]
    [tfms.update({k:ContTransformerLogRange}) for k in ["timetrain", "timepredict", "ramtrain", "rammodel", "rampredict", "ias"]]
    [tfms.update({k:ContTransformerLog2Range}) for k in ["ranger.num.trees", "ranger.min.node.size", 'ranger.num.random.splits', "rpart.cp", "glmnet.s", "xgboost.nrounds", "xgboost.eta", "xgboost.gamma", "xgboost.lambda", "xgboost.alpha", "xgboost.min_child_weight"]]
    [tfms.update({k:ContTransformerNegExpRange}) for k in ["logloss"]]
    return fit_config(key, tfms=tfms, **kwargs)


if __name__ == '__main__':
    wandb.login()
    #device = torch.device("cpu")
    device = 'cuda:0'
    # general tuning example workflow
    # 1. specify transformers you want to consider fixed (and not tuned over)
    tfms_xgboost = {}
    tfms_xgboost.update({"nf":tfms_chain([ContTransformerInt, ContTransformerRange])})
    # 2. tune by providing the fixed transformers (if any)
    study_xgboost = tune_config("iaml_xgboost", name="tune_iaml_xgboost_new", tfms_fixed=tfms_xgboost)
    # 3. extract the best params and refit the model (FIXME: could refit on whole train + valid data?)
    #    set export = True so that the onnx model is exported
    #    caveat: this overwrites the exiting model! # FIXME: should versionize this automatically
    l = fit_from_best_params("iaml_xgboost", best_params=study_xgboost.best_params, tfms_fixed=tfms_xgboost, export=True, device=device, epochs=300)
    # 4. get the performance metrics on the test set relying on the newly exported onnx model
    get_testset_metrics("iaml_xgboost")

    # load existing one:
    name = storage_name = "sqlite:///{}.db".format("tune_iaml_xgboost_new")
    study_xgboost = optuna.load_study(None, storage = name)

    # tfms_super nf tfms_chain([ContTransformerInt, ContTransformerRange]
    tfms_super = {}
    tfms_super.update({"nf":tfms_chain([ContTransformerInt, ContTransformerRange])})
    tfms_super.update({"ias":ContTransformerLogRange})
    params_super = copy(study_super.best_params)
    params_super.update({"tfms_logloss":"tlog"})
    fit_from_best_params("iaml_super", params_super, tfms_fixed=tfms_super, log_wandb=False, export=True, epochs=300)

