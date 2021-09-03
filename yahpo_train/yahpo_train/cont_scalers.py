import torch
import torch.nn as nn

class ContTransformerRange(nn.Module):
    """
    Transformer for Continuous Variables.
    Transforms to [p,1-p].
    """
    def __init__(self, x, p=0.01):
        super().__init__()
        self.p = torch.as_tensor(p)
        self.min, self.max = torch.min(x), torch.max(x)
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


class ContTransformerLogRange(nn.Module):
    """
    Log-Transformer for Continuous Variables.
    Transforms to [p,1-p] after applying a log-transform

    logfun :: Can be torch.log
    """
    def __init__(self, x, logfun = torch.log, expfun = torch.exp, p=0.01):
        super().__init__()
        self.p = torch.as_tensor(p)
        self.logfun = logfun
        self.expfun = self.expfun

        x = self.logfun(x)
        self.min, self.max = torch.min(x), torch.max(x)
        if self.max == self.min:
            raise Exception("Constant feature detected!")
        

    def forward(self, x):
        """
        Batch-wise transform for x
        """
        x = self.logfun(x)
        x = (x - self.min) / ((self.max - self.min) / (1. - 2*self.p)) + self.p
        return x.float()

    def invert(self, x):
        """
        Batch-wise inverse transform for x
        """
        x = (x - self.p) * ((self.max - self.min) / (1. - 2*self.p)) + self.min
        x = self.expfun(x)
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
        self.q1, self.q0 = torch.quantile(x, q), torch.quantile(x, 1.-q)


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
