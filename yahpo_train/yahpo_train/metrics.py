from fastai.callback.wandb import WandbCallback
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from fastai.tabular.all import *
import wandb

class AvgTfedMetric(Metric):
    """
    Average the values of `func` taking into account potential different batch sizes.
    Specializes to "transformed metrics saved during get_one_batch
    """
    def __init__(self, func):  self.func = func
    def reset(self):           self.total,self.count = 0.,0
    def accumulate(self, learn):
        bs = find_bs(learn.tfyb)
        self.total += learn.to_detach(self.func(*learn.tfyb, learn.tfpred))*bs
        self.count += bs
        
    @property
    def value(self): return self.total/self.count if self.count != 0 else None

    @property
    def name(self):  return self.func.func.__name__ if hasattr(self.func, 'func') else  self.func.__name__

def mae(x,y,impute_nan=True):
    if impute_nan:
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)
    return mean_absolute_error(x.cpu(),y.cpu(), multioutput="raw_values")

def r2(x,y,impute_nan=True):
    if impute_nan:
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)
    return r2_score(x.cpu(),y.cpu(), multioutput="raw_values") 

def spearman(x,y,impute_nan=True):
    if impute_nan:
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)
    
    # Return 0.5 for constant batches
    if torch.all(y == y[1]) or torch.all(x == x[1]):
        x = np.array(x.cpu())
        y = np.array(y.cpu())
        return np.array([0.5 for _,_ in zip(np.rollaxis(x, 1), np.rollaxis(y, 1))])

    x = np.array(x.cpu())
    y = np.array(y.cpu())
    
    rho = [spearmanr(xs, ys)[0] if not ((xs[0] == xs).all() or (ys[0] == ys).all()) else 0. for xs,ys in zip(np.rollaxis(x, 1), np.rollaxis(y, 1)) ]
    return np.array(rho)

def napct(x,y,impute_nan=True):
    return torch.mean(torch.isnan(y).float())

class WandbMetricsTableCallback(WandbCallback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def after_epoch(self):
        "Log validation loss and custom metrics & log prediction samples"
        # Correct any epoch rounding error and overwrite value
        self._wandb_epoch = round(self._wandb_epoch)
        wandb.log({'epoch': self._wandb_epoch}, step=self._wandb_step)
        # Log sample predictions
        if self.log_preds:
            try:
                self.log_predictions(self.learn.fetch_preds.preds)
            except Exception as e:
                self.log_preds = False
                self.remove_cb(FetchPredsCallback)
                print(f'WandbCallback was not able to get prediction samples -> {e}')
        
        log_dict = {}
        for n,s in zip(self.recorder.metric_names, self.recorder.log):
            if n not in ['train_loss', 'epoch', 'time']:
                if hasattr(s, "__len__"):
                    for m, nm in zip(s, self.recorder.dls.y_names):
                        log_dict.update({n+nm:m})
                else:
                    log_dict.update({n:s})

        wandb.log(log_dict, step=self._wandb_step)
