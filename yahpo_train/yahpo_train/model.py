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
        procs = [Categorify, FillMissing]
    )
    return dls

class ContNormalization(nn.Module):
    """
    Yeoh-Johnson Transformer
    Learns the transformation during initialization and
    outputs the transformation afterwards.
    """
    def __init__(self, x_sample, eps=.001, normalize=True, sigmoid_p = .01):
        super(ContNormalization, self).__init__()
        self.eps = eps
        self.normalize, self.sigmoid_p = normalize, sigmoid_p
        self.lmbda  = self.est_params(x_sample)
        xt = self.trafo_yj(x_sample, self.lmbda)
        if self.normalize:
            self.sigma, self.mu = torch.var_mean(xt, unbiased=True)
        else:
            self.min, self.max = torch.range(xt)
        

    def forward(self, x):
        x = self.trafo_yj(x, self.lmbda)
        if self.normalize:
            x = (x - self.mu) / torch.sqrt(self.sigma)
        else:
            x = (x - self.min) / ((self.max - self.min) / (1. - 2*self.sigmoid_p)) + self.sigmoid_p
        return x

    def invert(self, x):
        if self.normalize:
            x = x  * torch.sqrt(self.sigma) + self.mu
        else:
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
        if torch.abs(lmbda - 2.) < self.eps:
            x = torch.log1p(-x)
        else:
            x = (torch.pow(-x + 1, 2. - lmbda) - 1.) / (2. - lmbda)
        return x

    def est_params(self, x_sample):
        def neg_loglik(lmbda):
            xt = self.trafo_yj(x_sample, torch.tensor(lmbda))
            xt_var, xt_mean = torch.var_mean(xt, unbiased=True)
            cst = torch.sum(torch.sign(xt) * torch.log1p(torch.abs(xt)))
            return 0.5 * xt.shape[0] * torch.log(xt_var) + (lmbda - 1.) * cst
        res = scipy.optimize.minimize_scalar(neg_loglik, bounds=(-10, 10), method='bounded', options = {'maxiter':2000})
        return torch.tensor(res.x)

    def inverse_trafo_yj(self, x, lmbda):
        return torch.where(x >= 0, self.inv_pos(x, lmbda), self.inv_neg(x, lmbda))

    def inv_pos(self, x, lmbda):
        if torch.abs(lmbda) < self.eps:
            x = torch.exp(x) - 1
        else:
            x = torch.pow(x * lmbda + 1, 1 / lmbda) - 1.
        return x
    def inv_neg(self, x,  lmbda):
        if torch.abs(lmbda - 2.) < self.eps:
            x = -torch.exp(x)+1
        else:
            x = 1 - torch.pow((2.-lmbda) * x + 1. , 1. / (2. - lmbda))
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

class FFSurrogate(nn.Module):
    def __init__(self, dls, config, emb_szs = None, layers = [200, 100], out_size = 8, use_bn = False, ps=[0., 0.], actns=["selu", "selu", 'sigmoid']):
        super(FFSurrogate, self).__init__()
        emb_szs = get_emb_sz(dls.train_ds, {} if emb_szs is None else emb_szs)
        self.embds_fct = nn.ModuleList([Embedding(ni, nf) for ni, nf in emb_szs])
        self.embds_dbl = nn.ModuleList([ContNormalization(torch.from_numpy(cont).float()) for name, cont in dls.all_cols.sample(n=10000)[dls.cont_names].iteritems()])
        # self.embds_tgt = 
        self.n_emb,self.n_cont = sum(e.embedding_dim for e in self.embds_fct), len(dls.cont_names)
        sizes = [n_emb + n_cont] + layers + [out_sz]
        actns = [act_cls for _ in range(len(sizes)-2)] + [None]
        _layers = [LinBnDrop(sizes[i], sizes[i+1], bn=use_bn and i!=len(actns)-1, p=p, act=a, lin_first=False)
                       for i,(p,a) in enumerate(zip(ps+[0.],actns))]
        self.layers = nn.Sequential(*_layers)
        
    def forward(self, x):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embds_dbl)]
            x = torch.cat(x, 1)
        if self.n_cont != 0:
            xd = [e(x_cont[:,i]) for i,e in enumerate(self.embds_fct)]
            xd = torch.cat(xd, 1)
            x = torch.cat(x, xd) if self.n_emb > 0 else xd
        return self.layers(x)



if __name__ == '__main__':

    file = '~/LRZ Sync+Share/multifidelity_data/lcbench/data.csv'
    dls = make_dataloader(file)
    # pd.read_csv(file)

    x = torch.randn(1000)
    ct = ContNormalization(x)
    xf = ct.forward(x)
    xb = ct.invert(xf)
    print(x[1:20])
    print(xb[1:20])



  
