import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from fastai.tabular.all import *
from yahpo_train.cont_normalization import ContNormalization
from yahpo_train.embed_helpers import *

def dl_from_config(config, bs=1024, skipinitialspace=True, save_encoding=True, frac=1., **kwargs):
    # We shuffle the DataFrame before handing it to the dataloader to ensure mixed batches
    # All relevant info is obtained from the 'config'
    dtypes = dict(zip(config.cat_names, ["object"] * len(config.cat_names)))
    df = pd.read_csv(config.get_path("dataset"), skipinitialspace=skipinitialspace,dtype=dtypes).sample(frac=frac).reset_index()
    df.reindex(columns=config.cat_names+config.cont_names+config.y_names)
    
    dls = TabularDataLoaders.from_df(
        df = df,
        path = config.config_path,
        y_names = config.y_names,
        cont_names = config.cont_names,
        cat_names = config.cat_names,
        procs = [Categorify, FillMissing(fill_strategy=FillStrategy.constant, add_col=True, fill_vals=0)],  # FIXME: FillMissing correct?
        valid_idx = _get_valid_idx(df, config),
        bs = bs,
        shuffle=True,
        **kwargs
    )

    # Save the encoding of categories
    encoding = {cat_name:dict(dls.classes[cat_name].o2i) for cat_name in config.cat_names}

    if (save_encoding):       
        with open(config.get_path("encoding"), 'w') as f:
            json.dump(encoding, fp=f, sort_keys=True)

    return dls

def _get_valid_idx(df, config, frac=.1, rng_seed=10):
    """
    Include or exclude blocks of hyperparameters with differing fidelity
    The goal here is to not sample from the dataframe randomly, but instead either keep a hyperparameter group
    or drop it. 
    (By group I mean one config trained e.g. at epochs 1, ..., 50 )..
    """
    # All hyperpars excluding fidelity params
    hpars = config.cont_names+config.cat_names
    [hpars.remove(fp) for fp in config.fidelity_params]

    # Speed up for larger number of hyperparameters by converting cats to int.
    # Otherwise groupby breaks
    cont_hpars = set(hpars).intersection(set(config.cat_names))
    df = df[hpars].copy()
    df[cont_hpars].fillna('_NA_')
    df = df.apply(lambda x: pd.factorize(x.astype('category'))[0], axis=0)
    
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
    def __init__(self, dls, emb_szs = None, layers = [400, 400], deeper = [400, 400, 400], wide = True, use_bn = False, ps=0.1, act_cls=nn.SELU(inplace=True), final_act = nn.Sigmoid(), lin_first=False, embds_dbl=None, embds_tgt=None):
        super().__init__()

        if not (len(layers) | len(deeper) | wide):
            raise Exception("One of layers, deeper or wide has to be set!")

        emb_szs = get_emb_sz(dls.train_ds, {} if emb_szs is None else emb_szs)
        self.embds_fct = nn.ModuleList([Embedding(ni, nf) for ni, nf in emb_szs])

        # Transform continuous variables and targets
        if embds_dbl is not None:
            self.embds_dbl = nn.ModuleList([f(torch.from_numpy(cont[1].values).float()) for cont, f in zip(dls.all_cols[dls.cont_names].iteritems(), embds_dbl)])
        else:
            self.embds_dbl = nn.ModuleList([ContNormalization(torch.from_numpy(cont.values).float(), clip_outliers=False) for name, cont in dls.all_cols[dls.cont_names].iteritems()])
        
        if embds_tgt is not None:
            self.embds_tgt = nn.ModuleList([f(torch.from_numpy(cont[1].values).float()) for cont, f in zip(dls.ys.iteritems(), embds_tgt)])
        else:
            self.embds_tgt = nn.ModuleList([ContNormalization(torch.from_numpy(cont.values).float(), normalize='range', clip_outliers=True) for name, cont in dls.ys.iteritems()])

        self.n_emb,self.n_cont = sum(e.embedding_dim for e in self.embds_fct), len(dls.cont_names)
        self.sizes = [self.n_emb + self.n_cont] + layers + [dls.ys.shape[1]]

        self.deep, self.deeper, self.wide = nn.Sequential(), nn.Sequential(), nn.Sequential()
        # Deep Part
        if len(layers):
            ps1 = [ps for i in layers]
            self.sizes = [self.n_emb + self.n_cont] + layers + [dls.ys.shape[1]]
            actns = [act_cls for _ in range(len(self.sizes)-2)] + [None]
            _layers_deep = [LinBnDrop(self.sizes[i], self.sizes[i+1], bn=use_bn and i!=len(actns)-1, p=p, act=a, lin_first=lin_first)
                        for i,(p,a) in enumerate(zip(ps1+[0.],actns))]
            self.deep = nn.Sequential(*_layers_deep)

        # Deeper part
        if len(deeper):
            ps2 = [ps for i in deeper]
            self.deeper_sizes = [self.n_emb + self.n_cont] + deeper + [dls.ys.shape[1]]
            deeper_actns = [act_cls for _ in range(len(deeper))] + [None]
            _layers_deeper = [LinBnDrop(self.deeper_sizes[i], self.deeper_sizes[i+1], bn=use_bn and i!=len(deeper_actns)-1, p=p, act=a, lin_first=lin_first) for i,(p,a) in enumerate(zip(ps2+[0.],deeper_actns))]
            self.deeper = nn.Sequential(*_layers_deeper)

        if wide:
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
        
        xs = torch.zeros(x.shape[0], self.sizes[-1])
        if len(self.wide):
            xs = xs.add(self.wide(x))
        if len(self.deep):
            xs = xs.add(self.deep(x))
        if len(self.deeper):
            xs = xs.add(self.deeper(x))

        y = self.final_act(xs)
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
        """
        Export model to an ONNX file. We can safely ignore tracing errors with respect to lambda since lambda will be constant during inference.
        """
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
    from yahpo_gym.configuration import cfg
    from yahpo_gym.benchmarks import lcbench
    cfg = cfg("lcbench")
    dls = dl_from_config(cfg)
    f = FFSurrogateModel(dls, layers=[512,512], deeper = [], lin_first=False)
    l = SurrogateTabularLearner(dls, f, loss_func=nn.MSELoss(reduction='mean'), metrics=nn.MSELoss)
    l.add_cb(MixHandler)
    l.fit_one_cycle(5, 1e-4)
    for p in l.model.wide.parameters():
        p.requires_grad = False
    l.fit_flat_cos(5, 1e-4)
    l.export_onnx(cfg)



  
