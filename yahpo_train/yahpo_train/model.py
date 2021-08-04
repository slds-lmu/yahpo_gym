from typing import MutableMapping
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.tabular.all import *
import scipy

def make_dataloader(file):
    y_names = ['time', 'val_accuracy', 'val_cross_entropy', 'val_balanced_accuracy', 'test_cross_entropy', 'test_balanced_accuracy']
    dls = TabularDataLoaders.from_csv(
        csv = file,
        y_names=y_names,
        cont_names = ['epoch', 'batch_size', 'learning_rate', 'momentum', 'weight_decay', 'num_layers', 'max_units', 'max_dropout'],
        cat_names = ['OpenML_task_id'],
        procs = [Categorify, FillMissing],
        valid_idx = [x for x in range(1000)],
        bs = 1024
    )
    return dls


def to_tensor(x):
    if torch.is_tensor(x):
        return x
    else:
        return torch.as_tensor(x, dtype=torch.float64)

class ContNormalization(nn.Module):
    """
    Yeoh-Johnson Transformer
    Learns the transformation during initialization and
    outputs the transformation afterwards.
    """
    def __init__(self, x_sample, lmbda = None, eps=1e-6, normalize='scale', sigmoid_p = .01):
        super(ContNormalization, self).__init__()
        self.eps = eps
        self.normalize, self.sigmoid_p = normalize, sigmoid_p
        if not lmbda:
            self.lmbda  = self.est_params(x_sample)
        else:
            self.lmbda = to_tensor(lmbda)
        xt = self.trafo_yj(x_sample, self.lmbda)

        if self.normalize == 'scale':
            self.sigma, self.mu = torch.var_mean(xt)
        elif self.normalize == 'range':
            self.min, self.max = torch.min(xt), torch.max(xt)
            
    def forward(self, x):
        x = self.trafo_yj(x.float(), self.lmbda)
        if self.normalize == 'scale':
            x = (x - self.mu) / torch.sqrt(self.sigma)
        elif self.normalize == 'range':
            x = (x - self.min) / ((self.max - self.min) / (1. - 2*self.sigmoid_p)) + self.sigmoid_p
        return x.unsqueeze(1)

    def invert(self, x):
        if self.normalize == 'scale':
            x = x  * torch.sqrt(self.sigma) + self.mu
        elif self.normalize == 'range':
            x = (x - self.sigmoid_p) * ((self.max - self.min) / (1. - 2*self.sigmoid_p)) + self.mi
        x = self.inverse_trafo_yj(x, self.lmbda) 
        return x

    def trafo_yj(self, x, lmbda):   
        return torch.where(x >= 0, self.scale_pos(x, lmbda), self.scale_neg(x, lmbda))

    def scale_pos(self, x, lmbda):
        if torch.abs(lmbda) < self.eps:
            x = torch.log1p(x)
        else:
            x = (torch.pow(x + 1., lmbda) - 1) / lmbda 
        return x
    def scale_neg(self, x, lmbda):
        if torch.abs(lmbda - 2.) <= self.eps:
            x = torch.log1p(-x)
        else:
            x = (torch.pow(-x + 1, 2. - lmbda) - 1.) / (2. - lmbda)
        return -x

    def _neg_loglik(self, lmbda, x_sample):
        xt = self.trafo_yj(x_sample, to_tensor([lmbda]))
        xt_var, _ = torch.var_mean(xt, dim=0, unbiased=False)
        loglik = - 0.5 * x_sample.shape[0] * torch.log(xt_var)
        loglik += (lmbda - 1.) * torch.sum(torch.sign(x_sample) * torch.log1p(torch.abs(x_sample)))
        return - loglik

    def est_params(self, x_sample):
        res = scipy.optimize.minimize_scalar(lambda x: self._neg_loglik(x, x_sample), bounds=(-10, 10), method='bounded')
        return to_tensor(res.x)

    def inverse_trafo_yj(self, x, lmbda):
        return torch.where(x >= 0, self.inv_pos(x, lmbda), self.inv_neg(x, lmbda))

    def inv_pos(self, x, lmbda):
        if torch.abs(lmbda) < self.eps:
            x = torch.expm1(x)
        else:
            x = torch.pow(x * lmbda + 1, 1 / lmbda) - 1.
        return x
    def inv_neg(self, x,  lmbda):
        if torch.abs(lmbda - 2.) < self.eps:
            x = -torch.exp(x)+1
        else:
            x = 1 - torch.pow(-(2.-lmbda) * x + 1. , 1. / (2. - lmbda))
        return x
def emb_sz_rule(n_cat):
    "Rule of thumb to pick embedding size corresponding to `n_cat`"
    return min(600, round(1.6 * n_cat**0.56))

def _one_emb_sz(classes, n, sz_dict=None):
    "Pick an embedding size for `n` depending on `classes` if not given in `sz_dict`."
    sz_dict = ifnone(sz_dict, {})
    n_cat = len(classes[n])
    sz = sz_dict.get(n, int(emb_sz_rule(n_cat)))  # rule of thumb
    return n_cat,sz

def get_emb_sz(to, sz_dict=None):
    "Get default embedding size from `TabularPreprocessor` `proc` or the ones in `sz_dict`"
    return [_one_emb_sz(to.classes, n, sz_dict) for n in to.cat_names]

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
        self.pred = self.model(*self.xb)
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

class FFSurrogateModel(nn.Module):
    def __init__(self, dls, emb_szs = None, layers = [200, 100], out_size = 8, use_bn = False, ps=[0., 0.], act_cls=nn.SELU(inplace=True), final_act = nn.Sigmoid()):
        super(FFSurrogateModel, self).__init__()
        emb_szs = get_emb_sz(dls.train_ds, {} if emb_szs is None else emb_szs)
        self.embds_fct = nn.ModuleList([Embedding(ni, nf) for ni, nf in emb_szs])
        self.embds_dbl = nn.ModuleList([ContNormalization(torch.from_numpy(cont.values).float(),) for name, cont in dls.all_cols.sample(100000)[dls.cont_names].iteritems()])
        self.embds_tgt = nn.ModuleList([ContNormalization(torch.from_numpy(cont.values).float(), normalize='range') for name, cont in dls.ys.sample(100000).iteritems()])
        self.n_emb,self.n_cont = sum(e.embedding_dim for e in self.embds_fct), len(dls.cont_names)
        sizes = [self.n_emb + self.n_cont] + layers + [dls.ys.shape[1]]
        actns = [act_cls for _ in range(len(sizes)-2)] + [final_act]
        _layers = [LinBnDrop(sizes[i], sizes[i+1], bn=use_bn and i!=len(actns)-1, p=p, act=a, lin_first=False)
                       for i,(p,a) in enumerate(zip(ps+[0.],actns))]
        self.deep = nn.Sequential(*_layers)
        self.wide = nn.Sequential(nn.Linear(sizes[0], sizes[-1]), nn.SELU())
        
    def forward(self, x_cat, x_cont=None):
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

    def predict(self, x_cat, x_cont=None):
        y = self(x_cat, x_cont)
        return self.inv_trafo_ys(y)




if __name__ == '__main__':
    file = '~/LRZ Sync+Share/multifidelity_data/lcbench/data.csv'
    dls = make_dataloader(file)
    ff = FFSurrogateModel(dls)
    l = SurrogateTabularLearner(dls, ff, metrics=nn.MSELoss)
    l.fit_one_cycle(5)




  
