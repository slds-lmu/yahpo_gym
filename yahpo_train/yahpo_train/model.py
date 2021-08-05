from typing import MutableMapping
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import random
from fastai.tabular.all import *

from yahpo_train.embed_helpers import *

def dl_from_config(config, bs=1024, skipinitialspace=True, **kwargs):
    df = pd.read_csv(config.get_path("dataset"), skipinitialspace=skipinitialspace)
    dls = TabularDataLoaders.from_df(
        df = df,
        path = config.get_path("dataset"),
        y_names=config.y_names,
        cont_names = config.cont_names,
        cat_names = config.cat_names,
        procs = [Categorify, FillMissing],
        valid_idx = get_valid_idx(df, config),
        bs = bs,
        **kwargs
    )
    return dls

def get_valid_idx(df, config, frac=.1, rng_seed=10):
    "Include or exclude blocks of hyperparameters with differing fidelity"
    # All hyperpars excluding fidelity params
    hpars = config.cont_names+config.cat_names
    [hpars.remove(fp) for fp in config.fidelity_params]
    random.seed(rng_seed)
    idx = pd.Int64Index([])
    for _, dfg in df.groupby(hpars):
        # Sample index blocks
        if random.random() <= frac:
            idx = idx.append(dfg.index)
    return idx

class SurrogateTabularLearner(Learner):
    "`Learner` for tabular data"
    def predict(self, row):
        "Predict on a Pandas Series"
        dl = self.dls.test_dl(row.to_frame().T)
        dl.dataset.conts = dl.dataset.conts.astype(np.float32)
        inp,preds,_,dec_preds = self.get_preds(dl=dl, with_input=True, with_decoded=True)
        b = (*tuplify(inp),*tuplify(dec_preds))
        full_dec = self.dls.decode(b)
        return full_dec, dec_preds[0], preds[0]
    
    def _split(self, b):
        i = getattr(self.dls, 'n_inp', 1 if len(b)==1 else len(b)-1)
        self.xb, self.yb = b[:i], [self.model.trafo_ys(*b[i:])]

    def _do_one_batch(self):
        self.pred = self.model(*self.xb, invert_ytrafo = False)
        self('after_pred')
        if len(self.yb):
            self.loss_grad = self.loss_func(self.pred, *self.yb)
            self.loss = self.loss_grad.clone()
        self('after_loss')
        if not self.training or not len(self.yb): return
        self('before_backward')
        self.loss_grad.backward()
        self._with_events(self.opt.step, 'step', CancelStepException)
        self.opt.zero_grad()
    
    def export_onnx(self, config):
        return self.model.export_onnx(config)

class FFSurrogateModel(nn.Module):
    def __init__(self, dls, emb_szs = None, layers = [200, 100], out_size = 8, use_bn = False, ps=[0., 0.], act_cls=nn.SELU(inplace=True), final_act = nn.Sigmoid()):
        super(FFSurrogateModel, self).__init__()
        emb_szs = get_emb_sz(dls.train_ds, {} if emb_szs is None else emb_szs)
        self.embds_fct = nn.ModuleList([Embedding(ni, nf) for ni, nf in emb_szs])
        self.embds_dbl = nn.ModuleList([ContNormalization(torch.from_numpy(cont.values).float(),) for name, cont in dls.all_cols.sample(100000)[dls.cont_names].iteritems()])
        self.embds_tgt = nn.ModuleList([ContNormalization(torch.from_numpy(cont.values).float(), normalize='range') for name, cont in dls.ys.sample(100000).iteritems()])
        self.n_emb,self.n_cont = sum(e.embedding_dim for e in self.embds_fct), len(dls.cont_names)
        self.sizes = [self.n_emb + self.n_cont] + layers + [dls.ys.shape[1]]
        actns = [act_cls for _ in range(len(self.sizes)-2)] + [final_act]
        _layers = [LinBnDrop(self.sizes[i], self.sizes[i+1], bn=use_bn and i!=len(actns)-1, p=p, act=a, lin_first=False)
                       for i,(p,a) in enumerate(zip(ps+[0.],actns))]
        self.deep = nn.Sequential(*_layers)
        self.wide = nn.Sequential(nn.Linear(self.sizes[0], self.sizes[-1]), nn.SELU())
        
    def forward(self, x_cat, x_cont=None, invert_ytrafo = True):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embds_fct)]
            x = torch.cat(x, 1)
        if self.n_cont != 0:
            xd = [e(x_cont[:,i]) for i,e in enumerate(self.embds_dbl)]
            xd = torch.cat(xd, 1)
            x = torch.cat([x, xd], 1) if self.n_emb > 0 else xd
        return self.deep(x).add(self.wide(x))
    
    def trafo_ys(self, ys):
        ys = [e(ys[:,i]) for i,e in enumerate(self.embds_tgt)]
        ys = torch.cat(ys, 1)
        return ys

    def inv_trafo_ys(self, ys):
        ys = [e.invert(ys[:,i]) for i,e in enumerate(self.embds_tgt)]
        ys = torch.cat(ys, 1)
        return ys

    # def predict(self, x_cat, x_cont=None):
    #     y = self(x_cat, x_cont)
    #     return self.inv_trafo_ys(y)
    
    def export_onnx(self, config):
        self.eval()
        torch.onnx.export(self,
            (torch.ones(1, len(config.cat_names), dtype=torch.int), {'x_cont': torch.randn(1, len(config.cont_names))}),
            config.get_path("model"),
            do_constant_folding=True,
            export_params=True,
            input_names=['x_cat', 'x_cont'],
            opset_version=12
        )





if __name__ == '__main__':
    from yahpo_train.cont_normalization import ContNormalization
    from yahpo_gym import cfg
    from yahpo_gym.benchmarks import lcbench
    cfg = cfg("lcbench")
    dls = dl_from_config(cfg)
    ff = FFSurrogateModel(dls)
    l = SurrogateTabularLearner(dls, ff, metrics=nn.MSELoss)
    l.fit_one_cycle(5)
    l.export_onnx(cfg)



  
