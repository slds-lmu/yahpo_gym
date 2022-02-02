import random
import pandas as pd
from fastai.tabular.all import *


def dl_from_config(config, bs=1024, skipinitialspace=True, save_df_test=True, save_encoding=False, nrows=None, frac=1., train_frac=.8, **kwargs):
    """
    Instantiate a pytorch dataloader from a YAHPO config
    """
    # We shuffle the DataFrame before handing it to the dataloader to ensure mixed batches
    # All relevant info is obtained from the 'config'
    dtypes = dict(zip(config.cat_names, ["object"] * len(config.cat_names)))
    dtypes.update(dict(zip(config.cont_names+config.y_names, ["float32"] * len(config.cont_names+config.y_names))))
    df = pd.read_csv(config.get_path("dataset"), skipinitialspace=skipinitialspace,dtype=dtypes, nrows=nrows).sample(frac=frac).reset_index()
    df.reindex(columns=config.cat_names+config.cont_names+config.y_names)
    # Get rid of irrelevant columns
    df = df[config.cat_names+config.cont_names+config.y_names]
    # Fill missing target with 0
    df[config.y_names] = df[config.y_names].fillna(0.)

    # If train_frac = 0.8 and validation frac is 0.25, we get 0.6/0.2/0.2 train/valid/test
    train_ids = _get_idx(df, config, frac = train_frac)  # training ids of size train_frac used for df_train; df_test is complimentary
    df_train = df[df.index.isin(train_ids)].reset_index()
    df_test = df[~df.index.isin(train_ids)].reset_index()

    if save_df_test:
        df_test.to_csv(config.get_path("test_dataset"), index=False)

    dls = TabularDataLoaders.from_df(
        df = df_train,
        path = config.config_path,
        y_names = config.y_names,
        cont_names = config.cont_names,
        cat_names = config.cat_names,
        procs = [Categorify, FillMissing(fill_strategy=FillStrategy.constant, add_col=False, fill_vals=dict((k, 0.) for k in config.cat_names+config.cont_names))],
        valid_idx = _get_idx(df_train, config=config, frac=.25),  # validation ids of size 0.25 taken from training ids
        bs = bs,
        shuffle=
        True,
        **kwargs
    )

    # Save the encoding of categories
    encoding = {cat_name:dict(dls.classes[cat_name].o2i) for cat_name in config.cat_names}

    if save_encoding:
        with open(config.get_path("encoding"), 'w') as f:
            json.dump(encoding, fp=f, sort_keys=True)

    return dls

def _get_idx(df, config, frac=.2, rng_seed=10):
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
    df = df_shrink(df)
    df = df.apply(lambda x: pd.factorize(x.astype('category'))[0], axis=0)
    if len(hpars) > 10:
        hpars = random.sample(hpars, k=10)
    
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

        # For the training loss we train on untransformed scale.
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
    
    def _end_cleanup(self): 
        self.dl,self.xb,self.yb,self.pred,self.loss,self.tfpred,self.tfyb = None,(None,),(None,),None,None,None,None

    def export_onnx(self, config, device, suffix=''):
        return self.model.export_onnx(config, device, suffix)


if __name__ == '__main__':
    import torch.nn as nn
    from yahpo_gym.configuration import cfg
    from yahpo_gym.benchmarks import fcnet
    from yahpo_train.models import FFSurrogateModel, ResNet
    cfg = cfg("fcnet")
    dls = dl_from_config(cfg)

    print('Resnet:')
    f = ResNet(dls)
    l = SurrogateTabularLearner(dls, f, loss_func=nn.MSELoss(reduction='mean'), metrics=nn.MSELoss)
    l.add_cb(MixHandler)
    l.fit_one_cycle(5, 1e-4)
    l.export_onnx(cfg, 'cuda:0', suffix='resnet')

    # print('Feed Forward:')
    # f = FFSurrogateModel(dls, layers=[512,512], deeper = [], lin_first=False)
    # l = SurrogateTabularLearner(dls, f, loss_func=nn.MSELoss(reduction='mean'), metrics=nn.MSELoss)
    # l.add_cb(MixHandler)
    # l.fit_one_cycle(5, 1e-4)
    # l.export_onnx(cfg, 'cuda:0', suffix='ff')