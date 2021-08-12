from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from fastai.tabular.all import *

class AvgTfedMetric(Metric):
    """
    Average the values of `func` taking into account potential different batch sizes.
    Specializes to "transformed metrics saved during get_one_batch
    """
    def __init__(self, func):  self.func = func
    def reset(self):           self.total,self.count = 0.,0
    def accumulate(self, learn):
        bs = find_bs(learn.tfyb)
        self.total += learn.to_detach(self.func(*learn.tfyb, torch.nan_to_num(learn.tfpred, nan = 0.0)))*bs
        self.count += bs
    @property
    def value(self): return self.total/self.count if self.count != 0 else None
    @property
    def name(self):  return self.func.func.__name__ if hasattr(self.func, 'func') else  self.func.__name__

def mae(x,y):
    return mean_absolute_error(x,y, multioutput="raw_values")

def r2(x,y):
    return r2_score(x,y, multioutput="raw_values") 

def spearman(x,y):
    x = np.array(x)
    y = np.array(y)
    rho = [spearmanr(xs, ys)[0]  for xs,ys in zip(np.rollaxis(x, 1), np.rollaxis(y, 1))]
    return np.array(rho)