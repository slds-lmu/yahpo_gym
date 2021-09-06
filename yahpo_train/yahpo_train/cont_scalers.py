import torch
import torch.nn as nn

from yahpo_train.cont_normalization import to_tensor

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
    Transforms to [p,1-p] after applying a log-transform

    logfun :: Can be torch.log
    """
    def __init__(self, x, p=0.01):
        self.p = torch.as_tensor(p)
        super().__init__()
        x = torch.expm1(-x)
        self.min, self.max = torch.min(x[~torch.isnan(x)]), torch.max(x[~torch.isnan(x)])
        if self.max == self.min:
            raise Exception("Constant feature detected!")
        
    def forward(self, x):
        """
        Batch-wise transform for x
        """
        x = torch.exp(-x)
        x = (x - self.min) / ((self.max - self.min) / (1. - 2*self.p)) + self.p
        return x.float()

    def invert(self, x):
        """
        Batch-wise inverse transform for x
        """
        x = (x - self.p) * ((self.max - self.min) / (1. - 2*self.p)) + self.min
        x = - torch.log(x)
        return x.float()


class ContTransformerLogRange(nn.Module):
    """
    Log-Transformer for Continuous Variables.
    Transforms to [p,1-p] after applying a log-transform

    logfun :: Can be torch.log
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
    Log-Transformer for Continuous Variables.
    Transforms to [p,1-p] after applying a log-transform

    trafo :: A univariate function
    inverse :: A univariate function
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
    Log-Transformer for Continuous Variables.
    Transforms to [p,1-p] after applying a log-transform

    trafo :: A univariate function
    inverse :: A univariate function
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

    q :: quantile to clip
    """
    def __init__(self, x, q = .995):
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
        for tf in self.tfms():
            x = tf.forward(x)
        return x

    def invert(self, x):
        """
        Chained batch-wise inverse transform for x
        """
        for tf in reversed(self.tfms()):
            x = tf.invert(x)
        return x.float()


def _float_power(base, exp):
    """
    This is currently problematic due to numerical imprecision. torch.float_power would solve the problem 
    but currently can not be converted to ONNX.
    """
    out = torch.pow(base.to(torch.double), exp.to(torch.double))
    return out.to(torch.float64)

def float_pow10(base):
    return _float_power(base, torch.as_tensor(10.))