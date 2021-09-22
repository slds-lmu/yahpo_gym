from yahpo_train.model  import *
from yahpo_train.metrics import *
from yahpo_train.cont_scalers import *
from yahpo_gym.benchmarks import lcbench, rbv2, nasbench_301, fcnet, taskset
from yahpo_gym.configuration import cfg
from fastai.callback.wandb import *
from functools import partial
import wandb

def fit_config(key, embds_dbl=None, embds_tgt=None, tfms=None, lr = 1e-4, epochs=25, deep=[512,512,256], deeper=[], dropout=0., wide=True, use_bn=False, frac=1.0, bs=2048, export=False, log_wandb=True, wandb_entity='mfsurrogates'):
    """
    Fit function with hyperparameters
    """
    cc = cfg(key)
    dls = dl_from_config(cc, bs=bs, frac=frac)

    # Construct embds from transforms. tfms overwrites emdbs_dbl, embds_tgt
    if tfms is not None:
        embds_dbl = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.all_cols[dls.cont_names].iteritems()]
        embds_tgt = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.ys.iteritems()]

    # Instantiate learner
    f = FFSurrogateModel(dls, layers=deep, deeper=deeper, ps=dropout, use_bn = use_bn, wide=wide, embds_dbl=embds_dbl, embds_tgt=embds_tgt)
    l = SurrogateTabularLearner(dls, f, loss_func=nn.MSELoss(reduction='mean'), metrics=nn.MSELoss)
    l.metrics = [AvgTfedMetric(mae),  AvgTfedMetric(r2), AvgTfedMetric(spearman)]
    l.add_cb(MixHandler)
    l.add_cb(EarlyStoppingCallback(patience=3))

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
        l.export_onnx(cc)


def fit_nb301(key = 'nb301', **kwargs):
    embds_dbl = [partial(ContTransformerMultScalar, m = 1/52)]
    embds_tgt = [partial(ContTransformerMultScalar, m = 1/100), ContTransformerRange]
    fit_config(key, embds_dbl=embds_dbl, embds_tgt=embds_tgt, **kwargs)


def fit_rbv2_super(key = 'rbv2_super', **kwargs):
    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ["mmce", "f1", "auc", "aknn.k", "aknn.M", "rpart.maxdepth", "rpart.minsplit", "rpart.minbucket", "xgboost.max_depth"]]
    [tfms.update({k:partial(ContTransformerLogRange)}) for k in ["timetrain", "timepredict", "svm.cost", "svm.gamma"]]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log2,  expfun=torch.exp2 )}) for k in ["glmnet.s", "rpart.cp", "aknn.ef", "aknn.ef_construction", "xgboost.nrounds", "xgboost.eta", "xgboost.gamma", "xgboost.lambda", "xgboost.alpha", "xgboost.min_child_weight", "ranger.num.trees", "ranger.min.node.size", 'ranger.num.random.splits']]
    [tfms.update({k:ContTransformerNegExpRange}) for k in ["logloss"]]

    fit_config(key, tfms=tfms, **kwargs)

 

def fit_rbv2_svm(key = 'rbv2_svm', **kwargs):
    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ["mmce", "f1", "auc"]]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log, expfun=torch.exp)}) for k in ["timetrain", "timepredict", "cost", "gamma"]]
    [tfms.update({k:ContTransformerNegExpRange}) for k in ["logloss"]]

    fit_config(key, tfms=tfms, **kwargs)



def fit_rbv2_xgboost(key = 'rbv2_xgboost', **kwargs):
    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ["mmce", "f1", "auc", "max_depth"]]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log, expfun=torch.exp)}) for k in ["timetrain", "timepredict"]]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log2,  expfun=torch.exp2 )}) for k in ["nrounds", "eta", "gamma", "lambda", "alpha", "min_child_weight"]]
    [tfms.update({k:ContTransformerNegExpRange}) for k in ["logloss"]]

    fit_config(key, tfms=tfms, **kwargs)


def fit_rbv2_ranger(key = 'rbv2_ranger', **kwargs):
    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ["mmce", "f1", "auc"]]
    [tfms.update({k:partial(ContTransformerLogRange)}) for k in ["timetrain", "timepredict"]]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log2,  expfun=torch.exp2 )}) for k in ["num.trees", "min.node.size", 'num.random.splits']]
    [tfms.update({k:ContTransformerNegExpRange}) for k in ["logloss"]]
    fit_config(key, tfms=tfms, **kwargs)


def fit_rbv2_rpart(key = 'rbv2_rpart', **kwargs):
    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ["mmce", "f1", "auc", "maxdepth", "minsplit", "minbucket"]]
    [tfms.update({k:partial(ContTransformerLogRange)}) for k in ["timetrain", "timepredict"]]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log2,  expfun=torch.exp2 )}) for k in ["cp"]]
    [tfms.update({k:ContTransformerNegExpRange}) for k in ["logloss"]]
    fit_config(key, tfms=tfms, **kwargs)

def fit_rbv2_glmnet(key = 'rbv2_glmnet', **kwargs):
    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ["mmce", "f1", "auc"]]
    [tfms.update({k:partial(ContTransformerLogRange)}) for k in ["timetrain", "timepredict",]]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log2,  expfun=torch.exp2 )}) for k in ["s"]]
    [tfms.update({k:ContTransformerNegExpRange}) for k in ["logloss"]]
    fit_config(key, tfms=tfms, **kwargs)


def fit_rbv2_aknn(key = 'rbv2_aknn', **kwargs):
    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ["mmce", "f1", "auc", "k", "M"]]
    [tfms.update({k:partial(ContTransformerLogRange)}) for k in ["timetrain", "timepredict"]]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log2,  expfun=torch.exp2 )}) for k in ["ef", "ef_construction"]]
    [tfms.update({k:ContTransformerNegExpRange}) for k in ["logloss"]]
    fit_config(key, tfms=tfms, **kwargs)


def fit_fcnet(key = 'fcnet', **kwargs):
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ["batch_size", "n_units_1", "n_units_2"]]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log, expfun=torch.exp)}) for k in ["init_lr", "runtime", "n_params"]]
    [tfms.update({k:partial(ContTransformerNegExpRange, q=.975)}) for k in ["valid_loss"]]
    fit_config(key, tfms=tfms, **kwargs)


def fit_lcbench(key='lcbench', **kwargs):
    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ["val_accuracy", "val_balanced_accuracy", "test_balanced_accuracy", "batch_size", "max_units"]]
    [tfms.update({k:partial(ContTransformerNegExpRange, q=1.)}) for k in ["val_cross_entropy", "test_cross_entropy", "time"]]
    fit_config(key, tfms=tfms, **kwargs)


def fit_taskset(key='taskset', **kwargs):
    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ['replication']]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log2,  expfun=torch.exp2 )}) for k in ["epoch"]]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log,  expfun=torch.exp)}) for k in ["learning_rate", 'beta1', 'beta2', 'epsilon', 'l1', 'l2', 'linear_decay', 'exponential_decay']]
    [tfms.update({k:partial(ContTransformerNegExpRange, q=.99)}) for k in ["train", "valid1", "valid2", "test"]]
    fit_config(key, tfms=tfms, **kwargs)


if __name__ == '__main__':
    wandb.login()
    # fit_nb301(dropout=.0) # Done
    # fit_rbv2_rpart()
    # fit_rbv2_super()
    # fit_rbv2_svm()
    # fit_rbv2_xgboost()
    # fit_lcbench(export=True)
    # fit_rbv2_ranger()    
    # fit_rbv2_glmnet()
    # fit_rbv2_aknn(export=True)
    # fit_fcnet()
    fit_taskset(export=True)