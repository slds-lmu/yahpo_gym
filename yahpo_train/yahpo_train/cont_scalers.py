import torch
import torch.nn as nn
import scipy

from functools import partial

class ContTransformerNone(nn.Module):
    """
    Transformer for Continuous Variables. Performs no transformation (default operation)
    """
    def __init__(self, x):
        super().__init__()

    def forward(self, x):
        """
        Batch-wise transform for x
        """
        return x.float()
    def invert(self, x):
        """
        Batch-wise inverse transform for x
        """
        return x.float()


class ContTransformerRange(nn.Module):
    """
    Transformer for Continuous Variables.
    Transforms to [p,1-p].
    """
    def __init__(self, x, p=0.01):
        super().__init__()
        self.p = torch.as_tensor(p)
        self.min, self.max = torch.min(x[~torch.isnan(x)]), torch.max(x[~torch.isnan(x)])
        if self.max == self.min:
            raise Exception("Constant feature detected!")
        

    def forward(self, x):
        """
        Batch-wise transform for x
        """
        x = (x - self.min) / ((self.max - self.min) / (1. - 2*self.p)) + self.p
        return x.float()

    def invert(self, x):
        """
        Batch-wise inverse transform for x
        """
        x = (x - self.p) * ((self.max - self.min) / (1. - 2*self.p)) + self.min
        return x.float()


class ContTransformerNegExpRange(nn.Module):
    """
    Log-Transformer for Continuous Variables.
    Transforms to [p,1-p] after applying a neg-exp transform
    """
    def __init__(self, x, p=0.01, scale=True, q=1.):
        self.p = torch.as_tensor(p)
        self.scale = scale
        super().__init__()
        
        self.max = torch.as_tensor(1.).to(torch.double)
        if scale:
            self.max = max(torch.quantile(x[~torch.isnan(x)],q=q).to(torch.double), self.max)

        x = x.to(torch.double) / self.max
        x = torch.exp(-x)
        self.min, self.max = torch.min(x[~torch.isnan(x)]), torch.max(x[~torch.isnan(x)])
        if self.max == self.min:
            raise Exception("Constant feature detected!")
        
    def forward(self, x):
        """
        Batch-wise transform for x: x -> exp(-x / scale) -> [p, 1-p]
        """
        x = x.to(torch.double) / self.max
        x = torch.exp(-x)
        x = (x - self.min) / ((self.max - self.min) / (1. - 2*self.p)) + self.p
        return x.float()

    def invert(self, x):
        """
        Batch-wise inverse transform for x
        """
        x = (x - self.p) * ((self.max - self.min) / (1. - 2*self.p)) + self.min
        x = - torch.log(x.to(torch.double))
        x = x * self.max
        return x.float()


class ContTransformerLogRange(nn.Module):
    """
    Log-Transformer for Continuous Variables.
    Transforms to [p,1-p] after applying a log-transform

    logfun :: Can be torch.log
    expfun :: Can be torch.exp
    """
    def __init__(self, x, logfun = torch.log, expfun = torch.exp, p=0.01, eps = 1e-8):
        self.p = torch.as_tensor(p)
        self.eps = torch.as_tensor(eps)
        self.logfun = logfun
        self.expfun = expfun
        super().__init__()

        x = self.logfun(x + self.eps)
        self.min, self.max = torch.min(x[~torch.isnan(x)]), torch.max(x[~torch.isnan(x)])
        if self.max == self.min:
            raise Exception("Constant feature detected!")
        
    def forward(self, x):
        """
        Batch-wise transform for x
        """
        x = self.logfun(x + self.eps)
        x = (x - self.min) / ((self.max - self.min) / (1. - 2*self.p)) + self.p
        return x.float()

    def invert(self, x):
        """
        Batch-wise inverse transform for x
        """
        x = (x - self.p) * ((self.max - self.min) / (1. - 2*self.p)) + self.min
        x = torch.clamp(self.expfun(x) - self.eps, min = 0.)
        return x.float()


class ContTransformerMultScalar(nn.Module):
    """
    Transformer for Continuous Variables.
    Transforms by multiplying with a scalar
    """
    def __init__(self, x, m = 1.):
        super().__init__()
        self.m = torch.as_tensor(m)

    def forward(self, x):
        """
        Batch-wise transform for x
        """
        x = x * self.m
        return x.float()

    def invert(self, x):
        """
        Batch-wise inverse transform for x
        """
        x = x / self.m
        return x.float()


class ContTransformerFun(nn.Module):
    """
    Custom Transformer for Continuous Variables.
    """
    def __init__(self, x, trafo = lambda x: x, inverse = lambda x: x):
        super().__init__()
        self.trafo = trafo
        self.inverse = inverse

    def forward(self, x):
        """
        Batch-wise transform for x
        """
        x = self.trafo(x)
        return x.float()

    def invert(self, x):
        """
        Batch-wise inverse transform for x
        """
        x = self.inverse(x)
        return x.float()


class ContTransformerClipOutliers(nn.Module):
    """
    Clips large and small values according to quantiles.

    q :: quantile to clip to. Clipping values that deviate by more than one IQR range
    """
    def __init__(self, x, q = .99):
        super().__init__()
        self.q = q
        self.q1, self.q0 = torch.quantile(x[~torch.isnan(x)], q), torch.quantile(x[~torch.isnan(x)], 1.-q)


    def forward(self, x):
        """
        Batch-wise transform for x. Clip values that are larger/smaller then a quantile + IQR (range between quantiles).
        """
        q0, q1 = self.q0, self.q1
        iqr = torch.abs(q1 - q0)
        x_sample = torch.where(x > q1 + iqr, q1 + iqr, x)
        x_sample = torch.where(x < q0 - iqr, q0 + iqr, x)
        return x_sample

    def invert(self, x):
        "Clipping has no inverse"
        return x


class ContTransformerChain(nn.Module):
    """
    Chained transformer Continuous Variables. Chains several transforms.
    During forward pass, transforms are applied according to the list order,
    during invert, the order is reversed.
    """
    def __init__(self, x, tfms):
        self.tfms = [tf(x) for tf in tfms]
        super().__init__()

    def forward(self, x):
        """
        Chained batch-wise transform for x 
        """
        for tf in self.tfms:
            x = tf.forward(x)
        return x

    def invert(self, x):
        """
        Chained batch-wise inverse transform for x
        """
        for tf in reversed(self.tfms):
            x = tf.invert(x)
        return x.float()


ContTransformerScaleNegExp = partial(ContTransformerChain,tfms = [partial(ContTransformerRange, p=.1), ContTransformerNegExpRange])

class ContNormalization(nn.Module):
    """
    Yeoh-Johnson Transformer
    Learns the transformation during initialization and
    outputs the transformation afterwards.
    """
    def __init__(self, x_sample, lmbda = None, eps=1e-6, normalize='scale', sigmoid_p = .005, clip_outliers=True, impute_nan=True):
        super().__init__()
        self.eps = eps
        self.normalize, self.sigmoid_p = normalize, torch.as_tensor(sigmoid_p)
        self.impute_nan = impute_nan
        self.clip_outliers = clip_outliers

        x_sample = x_sample.double()
        
        # Deal with outliers and NaN
        if self.impute_nan:
            self.impute_val = None
            x_sample = self._impute_nan(x_sample)
        if self.clip_outliers:
            x_sample = self._clip_outliers(x_sample)

        # YJ Trafo train
        if not lmbda:
            self.lmbda = self.est_params(x_sample)
        else:
            self.lmbda = torch.as_tensor(lmbda)

        xt = self.trafo_yj(x_sample, self.lmbda)

        # Rescaling
        if self.normalize == 'scale':
            self.sigma, self.mu = torch.var_mean(xt)
            if self.sigma == 0.:
                raise Exception("Constant feature detected!")
        elif self.normalize == 'range':
            self.min, self.max = torch.min(xt), torch.max(xt)
            if self.max == self.min:
                raise Exception("Constant feature detected!")
            
    def forward(self, x): 
        x = x.double()
        if self.impute_nan:
            x = self._impute_nan(x)
        if self.clip_outliers:
            x = self._clip_outliers(x)

        x = self.trafo_yj(x, self.lmbda)

        if self.normalize == 'scale':
            x = (x - self.mu) / torch.sqrt(self.sigma)
        elif self.normalize == 'range':
            x = (x - self.min) / ((self.max - self.min) / (1. - 2*self.sigmoid_p)) + self.sigmoid_p
        return x.float()

    def invert(self, x):
        x = x.double()
        if self.normalize == 'scale':
            x = x * torch.sqrt(self.sigma) + self.mu
        elif self.normalize == 'range':
            x = (x - self.sigmoid_p) * ((self.max - self.min) / (1. - 2*self.sigmoid_p)) + self.min
          
        x = self.inverse_trafo_yj(x, self.lmbda)
        return x

    def trafo_yj(self, x, lmbda):
        return torch.where(x >= torch.as_tensor(0, dtype=torch.float64), self.scale_pos(x, lmbda), self.scale_neg(x, lmbda))

    def scale_pos(self, x, lmbda):
        if torch.abs(lmbda).numpy() < self.eps:
            x = torch.log1p(x)
        else:
            x = (_float_power(x + 1., lmbda) - 1) / lmbda 
        return x
    def scale_neg(self, x, lmbda):
        if torch.abs(lmbda - 2.).numpy() <= self.eps:
            x = torch.log1p(-x)
        else:
            x = (_float_power(-x + 1, 2. - lmbda) - 1.) / (2. - lmbda)
        return -x

    def _neg_loglik(self, lmbda, x_sample):
        """
        Negative Log-Likelihood optimized inside Yeo-Johnson transform 
        """
        xt = self.trafo_yj(x_sample, torch.as_tensor(lmbda).double())
        xt_var, _ = torch.var_mean(xt, dim=0, unbiased=False)
        loglik = - 0.5 * x_sample.shape[0] * torch.log(xt_var)
        loglik += (lmbda - 1.) * torch.sum(torch.sign(x_sample) * torch.log1p(torch.abs(x_sample)))
        return - loglik

    def est_params(self, x_sample):
        # res = optimize.minimize_scalar(lambda lmbd: self._neg_loglik(lmbd, x_sample), bracket=(-2.001, 2.001), method='brent', tol = 1e-8, options={'maxiter': 1000}).x
        _, lmbda = scipy.stats.yeojohnson(x_sample, lmbda=None)
       
        return torch.tensor(lmbda)

    def inverse_trafo_yj(self, x, lmbda):
        """
        The inverse trafo is not defined e.g., case one: x = 0.9, lmbda=-1.5, then the 'inv_pos' part is not defined.
        We should perhaps figure out what this translates to.
        """
        return torch.where(x >= torch.as_tensor(0, dtype=torch.float64), self.inv_pos(x, lmbda), self.inv_neg(x, lmbda))

    def inv_pos(self, x, lmbda):
        if torch.abs(lmbda) <= self.eps:
            x = torch.expm1(x)
        else:
            x = _float_power(x * lmbda + 1., 1. / lmbda) - 1.
        return x
    
    def inv_neg(self, x, lmbda):
        if torch.abs(lmbda - 2.) < self.eps:
            #x = -torch.exp(x)+1.
            x = -torch.expm1(-x)
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
        if self.impute_val is None:
            self.impute_val = torch.mode(x_sample[~torch.isnan(x_sample)]).values
        x_sample = torch.nan_to_num(x_sample, self.impute_val)
        return x_sample


def _float_power(base, exp):
    """
    This is currently problematic due to numerical imprecision. torch.float_power would solve the problem 
    but currently can not be converted to ONNX.
    """
    out = torch.pow(base.to(torch.double), exp.to(torch.double))
    return out.to(torch.float64)

def float_pow10(base):
    return _float_power(base, torch.as_tensor(10.))
