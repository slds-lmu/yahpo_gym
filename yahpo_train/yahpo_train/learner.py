import random
import pandas as pd
import numpy as np
from fastai.tabular.all import *
from yahpo_gym.configuration import Configuration


def dl_from_config(
    config: "yahpo_gym.configuration.Configuration",
    bs: int = None,
    skipinitialspace: bool = True,
    save_df_test: bool = True,
    save_encoding: bool = False,
    train_frac: float = 0.8,
    valid_frac: float = 0.25,
    rng_seed: int = 10,
    **kwargs
) -> "fastai.tabular.data.TabularDataLoaders":
    """Create a fastai dataloader from a config file."""
    np.random.seed(rng_seed)
    random.seed(rng_seed)

    # we shuffle the DataFrame before handing it to the dataloader to ensure mixed batches
    # all relevant info is obtained from the 'config'
    dtypes = dict(zip(config.cat_names, ["object"] * len(config.cat_names)))
    dtypes.update(
        dict(
            zip(
                config.cont_names + config.y_names,
                ["float32"] * len(config.cont_names + config.y_names),
            )
        )
    )
    df = pd.read_csv(
        config.get_path("dataset"),
        skipinitialspace=skipinitialspace,
        usecols=list(dtypes.keys()),
        dtype=dtypes,
    ).sample(frac=1.0)
    df.reindex(columns=config.cat_names + config.cont_names + config.y_names)
    # get rid of irrelevant columns
    df = df[config.cat_names + config.cont_names + config.y_names]
    # fill missing target with 0
    df[config.y_names] = df[config.y_names].fillna(0.0)

    # if train_frac = 0.8 and validation frac is 0.25, we get 0.8/0.1/0.1 train/valid/test
    test_ids = _get_idx(df, config, frac=1 - train_frac, rng_seed=rng_seed)
    df_test = df[df.index.isin(test_ids)].reset_index(drop=True)
    df_train = df[~df.index.isin(test_ids)].reset_index(drop=True)

    if save_df_test:
        df_test.to_csv(config.get_path("test_dataset"), index=False)

    if bs is None:
        # batch size is 2^x, where x is the smallest integer such that 2^x > len(df_train) * (1 - valid_frac)
        bs = 2 ** (int(math.log2((len(df_train) * (1 - valid_frac) / 100))) + 1)

    dls = TabularDataLoaders.from_df(
        df=df_train,
        path=config.config_path,
        y_names=config.y_names,
        cont_names=config.cont_names,
        cat_names=config.cat_names,
        procs=[
            Categorify,
            FillMissing(
                fill_strategy=FillStrategy.constant,
                add_col=False,
                fill_vals=dict((k, 0.0) for k in config.cont_names + config.cat_names),
            ),
        ],
        valid_idx=_get_idx(
            df_train, config=config, frac=valid_frac, rng_seed=rng_seed
        ),  # validation ids of size 0.25 * train_frac taken from train_ids
        bs=bs,
        shuffle=True,
        **kwargs
    )

    # save the encoding of categories
    encoding = {
        cat_name: dict(dls.classes[cat_name].o2i) for cat_name in config.cat_names
    }

    if save_encoding:
        with open(config.get_path("encoding"), "w") as f:
            json.dump(encoding, fp=f, sort_keys=True)

    return dls


def _get_idx(
    df: "pd.core.frame.DataFrame",
    config: "yahpo_gym.configuration.Configuration",
    frac: float = 0.25,
    rng_seed: int = 10,
    k: int = 10,
):
    """
    Include or exclude blocks of hyperparameters with differing fidelity.
    """
    np.random.seed(rng_seed)
    random.seed(rng_seed)
    hpars = config.cont_names + config.cat_names
    hpars = list(set(hpars) - set(config.fidelity_params))

    # speed up for larger number of hyperparameters by converting cats to int
    # otherwise groupby breaks
    cat_hpars = list(set(hpars).intersection(set(config.cat_names)))
    df = df[hpars].copy()
    df[cat_hpars].fillna("_NA_")
    df = df_shrink(df)
    df = df.apply(lambda hpar: pd.factorize(hpar.astype("category"))[0], axis=0)
    if len(hpars) > 10:
        hpars = random.sample(hpars, k=10)

    idx = pd.Index([], dtype="int64")
    groups = df.groupby(hpars, sort=False)
    for _, dfg in groups:
        # sample index blocks
        if random.random() <= frac:
            idx = idx.append(dfg.index)
    return idx


class SurrogateTabularLearner(Learner):
    """Learner for tabular data"""

    def _do_one_batch(self):
        if not self.training:
            self.tfpred = self.model(*self.xb, invert_ytrafo=True)
            self.tfyb = self.yb

        # for the training loss we train on untransformed scale
        self.pred = self.model(*self.xb, invert_ytrafo=False)
        self.yb = [self.model.trafo_ys(*self.yb)]

        self("after_pred")
        if len(self.yb):
            self.loss_grad = self.loss_func(self.pred, *self.yb)
            self.loss = self.loss_grad.clone()

        self("after_loss")
        if not self.training or not len(self.yb):
            return

        self("before_backward")
        self.loss_grad.backward()
        self._with_events(self.opt.step, "step", CancelStepException)
        self.opt.zero_grad()

    def _end_cleanup(self):
        self.dl, self.xb, self.yb, self.pred, self.loss, self.tfpred, self.tfyb = (
            None,
            (None,),
            (None,),
            None,
            None,
            None,
            None,
        )

    def export_onnx(self, config, device, suffix=""):
        return self.model.export_onnx(config, device, suffix)
