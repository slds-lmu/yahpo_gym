import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import random
from fastai.tabular.all import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from yahpo_train.cont_normalization import ContNormalization
from yahpo_train.embed_helpers import *

def dl_from_config(config, bs=1024, skipinitialspace=True, **kwargs):
    df = pd.read_csv(config.get_path("dataset"), skipinitialspace=skipinitialspace).sample(frac=1)
    df.reindex(columns=config.cat_names+config.cont_names+config.y_names)
    dls = TabularDataLoaders.from_df(
        df = df,
        path = config.config_path,
        y_names=config.y_names,
        cont_names = config.cont_names,
        cat_names = config.cat_names,
        procs = [Categorify, FillMissing],
        valid_idx = get_valid_idx(df, config),
        bs = bs,
        shuffle=True,
        **kwargs
    )
    return dls

def get_valid_idx(df, config, frac=.05, rng_seed=10):
    "Include or exclude blocks of hyperparameters with differing fidelity"
    # All hyperpars excluding fidelity params
    hpars = config.cont_names+config.cat_names
    [hpars.remove(fp) for fp in config.fidelity_params]
    # random.seed(rng_seed)
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
    
    def _do_one_batch(self):
        # Log transformed predictions and untransformed yb (tf is on the original scale)
        if not self.training: 
            self.tfpred = self.model(*self.xb, invert_ytrafo = True)
            self.tfyb = self.yb
        self.pred = self.model(*self.xb, invert_ytrafo = False)
        self.yb = [self.model.trafo_ys(*self.yb)]
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
    
    def _end_cleanup(self): self.dl,self.xb,self.yb,self.pred,self.loss, self.tfpred, self.tfyb = None,(None,),(None,),None,None,None,None

    def export_onnx(self, config):
        return self.model.export_onnx(config)

class FFSurrogateModel(nn.Module):
    def __init__(self, dls, emb_szs = None, layers = [400, 400], deeper = [400, 400, 400], out_size = 8, use_bn = False, ps=[0.1, 0.1], act_cls=nn.SELU(inplace=True), final_act = nn.Sigmoid()):
        super(FFSurrogateModel, self).__init__()
        emb_szs = get_emb_sz(dls.train_ds, {} if emb_szs is None else emb_szs)
        self.embds_fct = nn.ModuleList([Embedding(ni, nf) for ni, nf in emb_szs])
        self.embds_dbl = nn.ModuleList([ContNormalization(torch.from_numpy(cont.values).float(),) for name, cont in dls.all_cols[dls.cont_names].iteritems()])
        self.embds_tgt = nn.ModuleList([ContNormalization(torch.from_numpy(cont.values).float(), normalize='range') for name, cont in dls.ys.iteritems()])
        self.n_emb,self.n_cont = sum(e.embedding_dim for e in self.embds_fct), len(dls.cont_names)
        self.sizes = [self.n_emb + self.n_cont] + layers + [dls.ys.shape[1]]
        actns = [act_cls for _ in range(len(self.sizes)-2)] + [None]
        _layers_deep = [LinBnDrop(self.sizes[i], self.sizes[i+1], bn=use_bn and i!=len(actns)-1, p=p, act=a, lin_first=False)
                       for i,(p,a) in enumerate(zip(ps+[0.],actns))]
        self.deep = nn.Sequential(*_layers_deep)

        # Deeper part
        if len(deeper):
            self.deeper_sizes = [self.n_emb + self.n_cont] + deeper + [dls.ys.shape[1]]
            deeper_actns = [act_cls for _ in range(len(deeper)-2)] + [None]
            ps = [ps[1] for x in deeper]
            _layers_deeper = [LinBnDrop(self.deeper_sizes[i], self.deeper_sizes[i+1], bn=use_bn and i!=len(deeper_actns)-1, p=p, act=a, lin_first=False)
                for i,(p,a) in enumerate(zip(ps+[0.],deeper_actns))]
            self.deeper = nn.Sequential(*_layers_deeper)
        
        self.wide = nn.Sequential(nn.Linear(self.sizes[0], self.sizes[-1]))
        self.final_act = final_act
        
    def forward(self, x_cat, x_cont=None, invert_ytrafo = True):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embds_fct)]
            x = torch.cat(x, 1)
        if self.n_cont != 0:
            xd = [e(x_cont[:,i]).unsqueeze(1) for i,e in enumerate(self.embds_dbl)]
            xd = torch.cat(xd, 1)
            x = torch.cat([x, xd], 1) if self.n_emb > 0 else xd
        x = self.deep(x).add(self.wide(x))
        # if self.has_deeper:
        #     x = x.add(self.deeper(x))
        y = self.final_act(x)
        if invert_ytrafo:
            return self.inv_trafo_ys(y)
        else:
            return y
            
    def trafo_ys(self, ys):
        ys = [e(ys[:,i]).unsqueeze(1) for i,e in enumerate(self.embds_tgt)]
        ys = torch.cat(ys, 1)
        return ys

    def inv_trafo_ys(self, ys):
        ys = [e.invert(ys[:,i]).unsqueeze(1) for i,e in enumerate(self.embds_tgt)]
        ys = torch.cat(ys, 1)
        return ys
    
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

class AvgTfedMetric(Metric):
    """
    Average the values of `func` taking into account potential different batch sizes.
    Specializes to "transformed metrics saved during get_one_batch
    """
    def __init__(self, func):  self.func = func
    def reset(self):           self.total,self.count = 0.,0
    def accumulate(self, learn):
        bs = find_bs(learn.tfyb)
        # print(torch.max(learn.pred))
        self.total += learn.to_detach(self.func(*learn.tfyb, learn.tfpred, multioutput="raw_values"))*bs
        self.count += bs
    @property
    def value(self): return self.total/self.count if self.count != 0 else None
    @property
    def name(self):  return self.func.func.__name__ if hasattr(self.func, 'func') else  self.func.__name__

if __name__ == '__main__':
    from yahpo_train.cont_normalization import ContNormalization
    from yahpo_gym import cfg
    from yahpo_gym.benchmarks import lcbench
    from torchsummary import summary
    cfg = cfg("lcbench")
    dls = dl_from_config(cfg)
    ff = FFSurrogateModel(dls)
    l = SurrogateTabularLearner(dls, ff, metrics=nn.MSELoss)
    l.metrics = AvgTfedMetric(mean_absolute_error)
    l.lr_find()
    l.fit_one_cycle(5)
    # l.export_onnx(cfg)



  
