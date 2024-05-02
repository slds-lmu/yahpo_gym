import gc
import json
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from fastai.tabular.all import CancelStepException, Categorify, FillMissing, FillStrategy, Learner, df_shrink
from fastai.tabular.data import TabularDataLoaders
from pandas.core.frame import DataFrame
from yahpo_gym.configuration import Configuration


def dl_from_config(
    config: Configuration,
    bs: Optional[int] = None,
    skipinitialspace: bool = True,
    save_df_test: bool = True,
    save_encoding: bool = False,
    train_frac: float = 0.8,
    valid_frac: float = 0.25,
    shuffle: bool = True,
    device: torch.device = torch.device("cpu"),
    seed: int = 10,
    **kwargs
) -> Tuple[TabularDataLoaders, TabularDataLoaders]:
    """
    Create a fastai train and refit dataloader from a Configuration object.
    """
    np.random.seed(seed)
    random.seed(seed)

    # we shuffle the DataFrame before handing it to the dataloader to ensure mixed batches
    # all relevant info is obtained from the config
    dtypes = dict(zip(config.cat_names, ["object"] * len(config.cat_names)))
    dtypes.update(
        dict(
            zip(
                config.cont_names + config.y_names,
                ["float32"] * len(config.cont_names + config.y_names),
            )
        )
    )

    chunksize = 10**5
    chunks = []
    for chunk in pd.read_csv(
        config.get_path("dataset"),
        skipinitialspace=skipinitialspace,
        usecols=list(dtypes.keys()),
        dtype=dtypes,
        chunksize=chunksize,
    ):
        chunks.append(chunk)
    df = pd.concat(chunks, axis=0)
    df = df.sample(frac=1.00, random_state=seed)

    gc.collect()

    # get rid of irrelevant columns
    # if config.instance_names is not None we can be sure that it is the first element of config.cat_names
    df = df[config.cat_names + config.cont_names + config.y_names]

    # fill missing target with 0
    for y_name in config.y_names:
        df[y_name].fillna(0.0, inplace=True)

    # if train_frac = 0.8 and validation frac is 0.25, we get 0.6/0.2/0.2 train/valid/test
    test_ids = _get_idx(df, config, frac=1 - train_frac, seed=seed)
    df_test = df[df.index.isin(test_ids)].reset_index(drop=True)
    df_train = df[~df.index.isin(test_ids)].reset_index(drop=True)

    if save_df_test:
        df_test.to_csv(config.get_path("test_dataset"), index=False)

    if bs is None:
        potential_batch_sizes = 2 ** np.arange(5, 17)
        desired_fraction = 0.01
        desired_samples = len(df_train) * (1 - valid_frac) * desired_fraction
        bs = min(potential_batch_sizes, key=lambda x: abs(x - desired_samples))

    dl_train = TabularDataLoaders.from_df(
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
        valid_idx=list(
            _get_idx(df_train, config=config, frac=valid_frac, seed=seed)
        ),  # validation ids of size 0.25 * train_frac taken from train_ids
        bs=bs,
        shuffle=shuffle,
        device=device,
        **kwargs
    )

    dl_refit = TabularDataLoaders.from_df(
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
        valid_idx=[],
        bs=bs,
        shuffle=shuffle,
        device=device,
        **kwargs
    )

    # save the encoding of categories
    encoding = {
        cat_name: dict(dl_train.classes[cat_name].o2i) for cat_name in config.cat_names
    }

    if save_encoding:
        with open(Path(config.get_path("encoding")), "w") as f:
            json.dump(encoding, fp=f, sort_keys=True)

    return dl_train, dl_refit


def _get_idx(
    df: DataFrame,
    config: Configuration,
    frac: float = 0.25,
    seed: int = 10,
) -> pd.Index:
    """
    Include or exclude blocks of hyperparameters with differing fidelity.
    """
    np.random.seed(seed)
    random.seed(seed)
    hpars = config.cont_names + config.cat_names
    hpars = list(set(hpars) - set(config.fidelity_params))

    # speed up for larger number of hyperparameters by converting cats to int
    # otherwise groupby breaks
    cat_hpars = list(set(hpars).intersection(set(config.cat_names)))
    df = df[hpars].copy()
    for cat_hpar in cat_hpars:
        df[cat_hpar].fillna("_NA_", inplace=True)
    df = df_shrink(df)
    df = df.apply(lambda hpar: pd.factorize(hpar.astype("category"))[0], axis=0)

    idx = pd.Index([], dtype="int64")
    groups = df.groupby(hpars, sort=False)
    for _, dfg in groups:
        # sample index blocks
        if random.random() <= frac:
            idx = idx.append(dfg.index)
    return idx


class SurrogateTabularLearner(Learner):
    """Learner for tabular data"""

    def _do_one_batch(self) -> None:
        """
        Do one batch of training/validation/prediction.
        """
        if not self.training:
            self.tfpred = self.model(*self.xb, invert_ytrafo=True)
            self.tfyb = self.yb

        # for the training loss we train on untransformed scale
        self.pred = self.model(*self.xb, invert_ytrafo=False)
        if self.instance_names is not None:
            current_device = self.yb[0].device
            group = self.xb[0][:, 0].to(current_device)
            self.yb = [self.model.trafo_ys(*self.yb, group=group)]
        else:
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

    def _end_cleanup(self) -> None:
        """
        Clean up after the end of a training/validation/prediction.
        """
        self.dl, self.xb, self.yb, self.pred, self.loss, self.tfpred, self.tfyb = (
            None,
            (None,),
            (None,),
            None,
            None,
            None,
            None,
        )

    def export_onnx(
        self, config: Configuration, device: torch.device, suffix: str = ""
    ) -> None:
        """
        Export the model to ONNX.
        """
        return self.model.export_onnx(config, device, suffix)
