from yahpo_train.cont_normalization import ContNormalization
from yahpo_train.model  import *
from yahpo_train.metrics import *
from yahpo_train.cont_scalers import *
from yahpo_gym.benchmarks import lcbench, rbv2, nasbench_301, fcnet
from yahpo_gym.configuration import cfg
from fastai.callback.wandb import *
from functools import partial
import wandb

def fit_nb301():
    cc = cfg('nb301')
    dls = dl_from_config(cc, bs=2048, frac=.05
    
    )

    embds_dbl = [partial(ContTransformerMultScalar, m = 1/52)]
    embds_tgt = [partial(ContTransformerMultScalar, m = 1/100), ContTransformerRange]
    
    f = FFSurrogateModel(dls, layers=[512,512], embds_dbl=embds_dbl, embds_tgt=embds_tgt)
    l = SurrogateTabularLearner(dls, f, loss_func=nn.MSELoss(reduction='mean'), metrics=nn.MSELoss)
    l.metrics = [AvgTfedMetric(mae),  AvgTfedMetric(r2), AvgTfedMetric(spearman)]
    l.add_cb(MixHandler)
    l.add_cb(EarlyStoppingCallback(patience=3))
    
    # WandB
    wandb.init(project='surrogates-nb301', entity='pfistfl')
    l.add_cb(WandbCallback())
    wandb.config.update({'cont_tf': embds_dbl, 'tgt_tf': embds_tgt})

    l.fit_flat_cos(5, 1e-3)
    l.fit_flat_cos(5, 1e-4)
    l.export_onnx(cc)

def fit_rbv2_super():
    cc = cfg('rbv2_super')
    dls = dl_from_config(cc, bs=2048)

    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ["mmce", "f1", "auc", "aknn.k", "aknn.M", "rpart.maxdepth", "rpart.minsplit", "rpart.minbucket", "xgboost.max_depth"]]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log10, expfun=float_pow10)}) for k in ["timetrain", "timepredict", "svm.cost", "svm.gamma"]]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log2,  expfun=torch.exp2 )}) for k in ["glmnet.s", "rpart.cp", "aknn.ef", "aknn.ef_construction", "xgboost.nrounds", "xgboost.eta", "xgboost.gamma", "xgboost.lambda", "xgboost.alpha", "xgboost.min_child_weight", "ranger.num.trees", "ranger.min.node.size", 'ranger.num.random.splits']]
    [tfms.update({k:ContTransformerNegExpRange}) for k in ["logloss"]]
    embds_dbl = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.all_cols[dls.cont_names].iteritems()]
    embds_tgt = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.ys.iteritems()]

    f = FFSurrogateModel(dls, layers=[512,512], embds_dbl=embds_dbl, embds_tgt=embds_tgt)
    l = SurrogateTabularLearner(dls, f, loss_func=nn.MSELoss(reduction='mean'), metrics=nn.MSELoss)
    l.metrics = [AvgTfedMetric(mae),  AvgTfedMetric(r2), AvgTfedMetric(spearman)]
    l.add_cb(MixHandler)
    l.add_cb(EarlyStoppingCallback(patience=3))

    # WandB
    wandb.init(project='surrogates-rbv2_super', entity='pfistfl')
    l.add_cb(WandbCallback())
    wandb.config.update({'cont_tf': embds_dbl, 'tgt_tf': embds_tgt})

    l.fit_flat_cos(5, 1e-3)
    l.fit_flat_cos(10, 1e-4)
    l.export_onnx(cc)

def fit_fcnet():
    cc = cfg('fcnet')
    dls = dl_from_config(cc, bs=2048)
    f = FFSurrogateModel(dls, layers=[512,512])
    l = SurrogateTabularLearner(dls, f, loss_func=nn.MSELoss(reduction='mean'), metrics=nn.MSELoss)
    l.metrics = [AvgTfedMetric(mae),  AvgTfedMetric(r2), AvgTfedMetric(spearman)]
    l.add_cb(MixHandler)
    l.add_cb(EarlyStoppingCallback(patience=3))

    # WandB
    wandb.init(project='surrogates-fcnet', entity='pfistfl')
    l.add_cb(WandbCallback())
    wandb.config.update({'cont_tf': None, 'tgt_tf': None})

    l.fit_flat_cos(5, 1e-3)
    l.fit_flat_cos(10, 1e-4)
    l.export_onnx(cc)

def fit_lcbench():
    cc = cfg('lcbench')
    dls = dl_from_config(cc, bs=2048)

    # Transforms
    tfms = {}
    [tfms.update({k:ContTransformerRange}) for k in ["val_accuracy", "val_balanced_accuracy", "test_balanced_accuracy", "batch_size", "max_units"]]
    [tfms.update({k:partial(ContTransformerLogRange, logfun=torch.log10, expfun=float_pow10)}) for k in ["val_cross_entropy", "test_cross_entropy", "time"]]
    embds_dbl = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.all_cols[dls.cont_names].iteritems()]
    embds_tgt = [tfms.get(name) if tfms.get(name) is not None else ContTransformerNone for name, cont in dls.ys.iteritems()]

    # Model
    f = FFSurrogateModel(dls, layers=[512,512], embds_dbl=embds_dbl, embds_tgt=embds_tgt)
    l = SurrogateTabularLearner(dls, f, loss_func=nn.MSELoss(reduction='mean'), metrics=nn.MSELoss)
    l.metrics = [AvgTfedMetric(mae),  AvgTfedMetric(r2), AvgTfedMetric(spearman)]
    l.add_cb(MixHandler)
    l.add_cb(EarlyStoppingCallback(patience=3))

    # WandB
    wandb.init(project='surrogates-lcbench', entity='pfistfl')
    l.add_cb(WandbCallback())
    wandb.config.update({'cont_tf': embds_dbl, 'tgt_tf': embds_tgt})

    # Fit
    l.fit_one_cycle(5, 1e-4)  
    for p in l.model.wide.parameters():
        p.requires_grad = False
    l.fit_flat_cos(10, 1e-4)
    l.export_onnx(cc)


if __name__ == '__main__':
    wandb.login()
    fit_rbv2_super()