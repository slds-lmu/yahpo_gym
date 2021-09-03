from yahpo_train.cont_normalization import ContNormalization
from yahpo_train.model  import *
from yahpo_train.metrics import *
from yahpo_train.cont_scalers import *
from yahpo_gym.benchmarks import lcbench, rbv2, nasbench_301
from yahpo_gym.configuration import cfg
from functools import partial

def fit_nb301():
    cc = cfg('nb301')
    dls = dl_from_config(cc, bs=2048)
    embds_dbl = [partial(ContTransformerMultScalar, m = 1/52)]
    embds_tgt = [partial(ContTransformerMultScalar, m = 1/100), ContTransformerRange]
    f = FFSurrogateModel(dls, layers=[512,512], embds_dbl=embds_dbl, embds_tgt=embds_tgt)
    l = SurrogateTabularLearner(dls, f, loss_func=nn.MSELoss(reduction='mean'), metrics=nn.MSELoss)
    l.metrics = [AvgTfedMetric(mae),  AvgTfedMetric(r2), AvgTfedMetric(spearman)]
    l.add_cb(MixHandler)
    l.add_cb(EarlyStoppingCallback(patience=3))
    l.fit_flat_cos(5, 1e-3)
    l.fit_flat_cos(5, 1e-4)
    l.export_onnx(cc)


if __name__ == '__main__':
    fit_nb301()