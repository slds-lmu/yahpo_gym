from yahpo_train.cont_normalization import ContNormalization
from yahpo_train.model  import *
from yahpo_train.metrics import *
from yahpo_train.cont_scalers import *
from yahpo_gym.benchmarks import lcbench, rbv2, nasbench_301, fcnet
from yahpo_gym.configuration import cfg
from fastai.callback.wandb import *
from functools import partial
import wandb

def init_wandb_learner(key, l, frac=1.0):
    wandb.init(project=key, entity='mfsurrogates')
    l.add_cb(WandbMetricsTableCallback())
    wandb.config.update({'cont_tf': l.embds_dbl, 'tgt_tf': l.embds_tgt, 'fraction': frac}, allow_val_change=True)

def init_learner(dls, embds_dbl, embds_tgt):
    f = FFSurrogateModel(dls, layers=[512,512], embds_dbl=embds_dbl, embds_tgt=embds_tgt)
    l = SurrogateTabularLearner(dls, f, loss_func=nn.MSELoss(reduction='mean'), metrics=nn.MSELoss)
    l.metrics = [AvgTfedMetric(mae),  AvgTfedMetric(r2), AvgTfedMetric(spearman)]
    l.add_cb(MixHandler)
    l.add_cb(EarlyStoppingCallback(patience=3))
    return l
    

def fit_nb301(key = 'nb301', frac=1.0):
    cc = cfg(key)
    dls = dl_from_config(cc, bs=2048, frac=frac)

    embds_dbl = [partial(ContTransformerMultScalar, m = 1/52)]
    embds_tgt = [partial(ContTransformerMultScalar, m = 1/100), ContTransformerRange]
    
    l = init_learner(dls, embds_dbl, embds_tgt)
    init_wandb_learner(key, l, frac)

    l.fit_flat_cos(10, 1e-4)
    l.export_onnx(cc)


def fit_rbv2_super(key = 'rbv2_super', frac=1.0):
    cc = cfg(key)
    dls = dl_from_config(cc, bs=2048,frac=frac)

    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ["mmce", "f1", "auc", "aknn.k", "aknn.M", "rpart.maxdepth", "rpart.minsplit", "rpart.minbucket", "xgboost.max_depth"]]
    [tfms.update({k:partial(ContTransformerLogRange)}) for k in ["timetrain", "timepredict", "svm.cost", "svm.gamma"]]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log2,  expfun=torch.exp2 )}) for k in ["glmnet.s", "rpart.cp", "aknn.ef", "aknn.ef_construction", "xgboost.nrounds", "xgboost.eta", "xgboost.gamma", "xgboost.lambda", "xgboost.alpha", "xgboost.min_child_weight", "ranger.num.trees", "ranger.min.node.size", 'ranger.num.random.splits']]
    [tfms.update({k:ContTransformerNegExpRange}) for k in ["logloss"]]
    embds_dbl = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.all_cols[dls.cont_names].iteritems()]
    embds_tgt = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.ys.iteritems()]

    l = init_learner(dls, embds_dbl, embds_tgt)
    init_wandb_learner(key, l, frac)
    l.fit_flat_cos(10, 1e-4)
    l.export_onnx(cc)


def fit_rbv2_svm(key = 'rbv2_svm', frac=1.0):
    cc = cfg(key)
    dls = dl_from_config(cc, bs=512, frac=frac)
    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ["mmce", "f1", "auc"]]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log, expfun=torch.exp)}) for k in ["timetrain", "timepredict", "cost", "gamma"]]
    [tfms.update({k:ContTransformerNegExpRange}) for k in ["logloss"]]
    embds_dbl = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.all_cols[dls.cont_names].iteritems()]
    embds_tgt = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.ys.iteritems()]
    l = init_learner(dls, embds_dbl, embds_tgt)
    init_wandb_learner(key, l, frac)
    l.fit_flat_cos(10, 1e-4)
    l.export_onnx(cc)


def fit_rbv2_xgboost(key = 'rbv2_xgboost', frac=1.0):
    cc = cfg(key)
    dls = dl_from_config(cc, bs=2048, frac=frac)

    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ["mmce", "f1", "auc", "max_depth"]]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log, expfun=torch.exp)}) for k in ["timetrain", "timepredict"]]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log2,  expfun=torch.exp2 )}) for k in ["nrounds", "eta", "gamma", "lambda", "alpha", "min_child_weight"]]
    [tfms.update({k:ContTransformerNegExpRange}) for k in ["logloss"]]
    embds_dbl = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.all_cols[dls.cont_names].iteritems()]
    embds_tgt = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.ys.iteritems()]

    l = init_learner(dls, embds_dbl, embds_tgt)
    init_wandb_learner(key, l, frac)

    l.fit_flat_cos(10, 1e-4)
    l.export_onnx(cc)


def fit_rbv2_ranger(key = 'rbv2_ranger', frac=1.0):
    cc = cfg(key)
    dls = dl_from_config(cc, bs=2048,frac=frac)

    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ["mmce", "f1", "auc"]]
    [tfms.update({k:partial(ContTransformerLogRange)}) for k in ["timetrain", "timepredict"]]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log2,  expfun=torch.exp2 )}) for k in ["num.trees", "min.node.size", 'num.random.splits']]
    [tfms.update({k:ContTransformerNegExpRange}) for k in ["logloss"]]
    embds_dbl = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.all_cols[dls.cont_names].iteritems()]
    embds_tgt = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.ys.iteritems()]

    l = init_learner(dls, embds_dbl, embds_tgt)
    init_wandb_learner(key, l, frac)
    l.fit_flat_cos(10, 1e-4)
    l.export_onnx(cc)


def fit_rbv2_rpart(key = 'rbv2_rpart', frac=1.0):
    cc = cfg(key)
    dls = dl_from_config(cc, bs=2048,frac=frac)

    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ["mmce", "f1", "auc", "maxdepth", "minsplit", "minbucket"]]
    [tfms.update({k:partial(ContTransformerLogRange)}) for k in ["timetrain", "timepredict"]]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log2,  expfun=torch.exp2 )}) for k in ["cp"]]
    [tfms.update({k:ContTransformerNegExpRange}) for k in ["logloss"]]
    embds_dbl = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.all_cols[dls.cont_names].iteritems()]
    embds_tgt = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.ys.iteritems()]

    l = init_learner(dls, embds_dbl, embds_tgt)
    init_wandb_learner(key, l, frac)
    l.fit_flat_cos(10, 1e-4)
    l.export_onnx(cc)

def fit_rbv2_glmnet(key = 'rbv2_glmnet', frac=1.0):
    cc = cfg(key)
    dls = dl_from_config(cc, bs=2048,frac=frac)

    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ["mmce", "f1", "auc"]]
    [tfms.update({k:partial(ContTransformerLogRange)}) for k in ["timetrain", "timepredict",]]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log2,  expfun=torch.exp2 )}) for k in ["s"]]
    [tfms.update({k:ContTransformerNegExpRange}) for k in ["logloss"]]
    embds_dbl = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.all_cols[dls.cont_names].iteritems()]
    embds_tgt = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.ys.iteritems()]

    l = init_learner(dls, embds_dbl, embds_tgt)
    init_wandb_learner(key, l, frac)
    l.fit_flat_cos(10, 1e-4)
    l.export_onnx(cc)

def fit_rbv2_aknn(key = 'rbv2_aknn', frac=1.0):
    cc = cfg(key)
    dls = dl_from_config(cc, bs=2048,frac=frac)

    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ["mmce", "f1", "auc", "k", "M"]]
    [tfms.update({k:partial(ContTransformerLogRange)}) for k in ["timetrain", "timepredict"]]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log2,  expfun=torch.exp2 )}) for k in ["ef", "ef_construction"]]
    [tfms.update({k:ContTransformerNegExpRange}) for k in ["logloss"]]
    embds_dbl = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.all_cols[dls.cont_names].iteritems()]
    embds_tgt = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.ys.iteritems()]

    l = init_learner(dls, embds_dbl, embds_tgt)
    init_wandb_learner(key, l, frac)
    l.fit_flat_cos(10, 1e-4)
    l.export_onnx(cc)

def fit_fcnet(key = 'fcnet', frac=1.0):
    cc = cfg(key)
    dls = dl_from_config(cc, bs=2048, frac=frac)

    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ["batch_size", "n_units_1", "n_units_2"]]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log, expfun=torch.exp)}) for k in ["init_lr", "runtime", "n_params"]]
    [tfms.update({k:ContTransformerNegExpRange}) for k in ["valid_loss"]]


    embds_dbl = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.all_cols[dls.cont_names].iteritems()]
    embds_tgt = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.ys.iteritems()]

    l = init_learner(dls, embds_dbl, embds_tgt)
    init_wandb_learner(key, l, frac)

    l.fit_flat_cos(10, 1e-4)
    l.export_onnx(cc)


def fit_lcbench(key='lcbench', frac=1.0):
    cc = cfg(key)
    dls = dl_from_config(cc, bs=2048, frac=frac)
    
    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ["val_accuracy", "val_balanced_accuracy", "test_balanced_accuracy", "batch_size", "max_units"]]
    [tfms.update({k:ContTransformerNegExpRange}) for k in ["val_cross_entropy", "test_cross_entropy", "time"]]
    embds_dbl = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.all_cols[dls.cont_names].iteritems()]
    embds_tgt = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.ys.iteritems()]

    l = init_learner(dls, embds_dbl, embds_tgt)
    init_wandb_learner(key, l, frac)
    # Fit
    l.fit_flat_cos(10, 1e-4)
    l.export_onnx(cc)


def fit_taskset(key='taskset', frac=1.0):
    cc = cfg(key)
    dls = dl_from_config(cc, bs=2048, frac=frac)

    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ['replication']]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log2,  expfun=torch.exp2 )}) for k in ["epoch"]]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log,  expfun=torch.exp)}) for k in ["learning_rate", 'beta1', 'beta2', 'epsilon', 'l1', 'l2', 'linear_decay', 'exponential_decay']]
    [tfms.update({k:ContTransformerNegExpRange}) for k in ["train", "valid1", "valid2", "test"]]
    embds_dbl = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.all_cols[dls.cont_names].iteritems()]
    embds_tgt = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.ys.iteritems()]

    l = init_learner(dls, embds_dbl, embds_tgt)
    init_wandb_learner(key, l, frac)

    # Fit
    l.fit_flat_cos(10, 1e-4)
    l.export_onnx(cc)


if __name__ == '__main__':
    wandb.login()
    # fit_rbv2_svm()
    # fit_rbv2_xgboost()
    fit_rbv2_super()
    fit_lcbench()
    fit_nb301()
    fit_rbv2_ranger()    
    fit_rbv2_rpart()
    fit_rbv2_glmnet()
    fit_rbv2_aknn()


