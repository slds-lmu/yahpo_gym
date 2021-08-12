import torch
import torch.nn as nn
from scipy import optimize

class ContNormalization(nn.Module):
    """
    Yeoh-Johnson Transformer
    Learns the transformation during initialization and
    outputs the transformation afterwards.
    """
    def __init__(self, x_sample, lmbda = None, eps=1e-6, normalize='scale', sigmoid_p = .005, clip_outliers=True, impute_nan=True):
        super(ContNormalization, self).__init__()
        self.eps = eps
        self.normalize, self.sigmoid_p = normalize, to_tensor(sigmoid_p)
        self.impute_nan = impute_nan
        self.clip_outliers = clip_outliers

        # Deal with outliers and NaN
        if self.impute_nan:
            x_sample = self._impute_nan(x_sample)
        if self.clip_outliers:
            x_sample = self._clip_outliers(x_sample)
        # Trafo
        if not lmbda:
            self.lmbda  = self.est_params(to_tensor(x_sample))
        else:
            self.lmbda = to_tensor(lmbda)
        xt = self.trafo_yj(x_sample, self.lmbda)
        # Rescaling
        if self.normalize == 'scale':
            self.sigma, self.mu = torch.var_mean(xt)
        elif self.normalize == 'range':
            self.min, self.max = torch.min(xt), torch.max(xt)
            
    def forward(self, x): 
        if self.impute_nan:
            x = self._impute_nan(x)
        if self.clip_outliers:
            x = self._clip_outliers(x)

        x = self.trafo_yj(x.double(), self.lmbda)

        if self.normalize == 'scale':
            x = (x - self.mu) / torch.sqrt(self.sigma)
        elif self.normalize == 'range':
            x = (x - self.min) / ((self.max - self.min) / (1. - 2*self.sigmoid_p)) + self.sigmoid_p
        return x.float()

    def invert(self, x):
        if self.normalize == 'scale':
            x = x  * torch.sqrt(self.sigma) + self.mu
        elif self.normalize == 'range':
            x = (x - self.sigmoid_p) * ((self.max - self.min) / (1. - 2*self.sigmoid_p)) + self.min
        x = self.inverse_trafo_yj(x.double(), self.lmbda) 
        return x

    def trafo_yj(self, x, lmbda):   
        return torch.where(x >= 0, self.scale_pos(x, lmbda), self.scale_neg(x, lmbda))

    def scale_pos(self, x, lmbda):
        if torch.abs(lmbda) < self.eps:
            x = torch.log1p(x)
        else:
            x = (_float_power(x + 1., lmbda) - 1) / lmbda 
        return x
    def scale_neg(self, x, lmbda):
        if torch.abs(lmbda - 2.) <= self.eps:
            x = torch.log1p(-x)
        else:
            x = (_float_power(-x + 1, 2. - lmbda) - 1.) / (2. - lmbda)
        return -x

    def _neg_loglik(self, lmbda, x_sample):
        """
        Negative Log-Likelihood optimized inside Yeo-Johnson transform 
        """
        xt = self.trafo_yj(x_sample, to_tensor([lmbda]))
        xt_var, _ = torch.var_mean(xt, dim=0, unbiased=False)
        loglik = - 0.5 * x_sample.shape[0] * torch.log(xt_var)
        loglik += (lmbda - 1.) * torch.sum(torch.sign(x_sample) * torch.log1p(torch.abs(x_sample)))
        return - loglik

    def est_params(self, x_sample):
        res = optimize.minimize_scalar(lambda lmbd: self._neg_loglik(lmbd, x_sample), bounds=(-5, 5), method='bounded')
        return to_tensor(res.x)

    def inverse_trafo_yj(self, x, lmbda):
        """
        The inverse trafo is not defined e.g., case one: x = 0.9, lmbda=-1.5, then the 'inv_pos' part is not defined.
        We should perhaps figure out what this translates to.
        """
        return torch.where(x >= 0, self.inv_pos(x, lmbda), self.inv_neg(x, lmbda))

    def inv_pos(self, x, lmbda):
        if torch.abs(lmbda) <= self.eps:
            x = torch.expm1(x)
        else:
            x = _float_power(x * lmbda + 1., 1. / lmbda) - 1.
        return x
    
    def inv_neg(self, x,  lmbda):
        if torch.abs(lmbda - 2.) < self.eps:
            x = -torch.exp(x)+1.
        else:
            x = 1. - _float_power(-(2.-lmbda) * x + 1., 1. / (2. - lmbda))
        return x
    
    def _clip_outliers(self, x_sample):
        """
        Clip values that are greater than some quantile by at least IQR (and vice versa for smaller.)
        """
        q = .995
        q1, q0 = torch.quantile(x_sample, q), torch.quantile(x_sample, 1.-q)
        iqr = torch.abs(q1 - q0)
        x_sample = torch.where(x_sample > q1 + iqr, q1, x_sample)
        x_sample = torch.where(x_sample < q0 - iqr, q0, x_sample)
        return x_sample

    def _impute_nan(self, x_sample):
        x_sample = torch.nan_to_num(x_sample)
        return x_sample


def to_tensor(x):
    if torch.is_tensor(x):
        return x
    else:
        return torch.as_tensor(x, dtype=torch.float64)

# FIXME: replace with torch.float_power once ONNX opset allows for this
def _float_power(base, exp):
    """
    This is currently problematic due to numerical imprecision. torch.float_power would solve the problem 
    but currently can not be converted to ONNX.
    """
    out = torch.pow(base.to(torch.double), exp.to(torch.double))
    return out.to(torch.float64)

