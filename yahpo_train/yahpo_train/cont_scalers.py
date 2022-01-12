import torch
import torch.nn as nn
import scipy

from functools import partial

# Available Scalers:
# ContTransformerNone :: No transform
# ContTransformerRange :: Scale to [p , 1-p]
# ContTransformerClamp :: Clamp at min, max
# ContTransformerNegExp :: exp(-x)
# ContTransformerLog :: log or log2
# ContTransformerMultScalar :: x * factor
# ContTransformerInt :: convert to int
# ContTransformerChain :: chain multiple transformers
# Available chains: 
#   ContTransformerLog2Range :: log2 -> range
#   ContTransformerLogRange :: log -> range
#   ContTransformerNegExpRange :: negexp -> range
#   ContTransformerClamp01Range :: clamp 0,1 -> range
#   ContTransformerClamp0LogRange ::  clamp 0,Inf -> range


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
    Transformer for Continuous Variables. Transforms to [0,1]
    """
    def __init__(self, x):
        super().__init__()
        self.min, self.max = torch.min(x[~torch.isnan(x)]), torch.max(x[~torch.isnan(x)])
        if self.max == self.min:
            raise Exception("Constant feature detected!")
        

    def forward(self, x):
        """
        Batch-wise transform for x
        """
        x = (x - self.min) / (self.max - self.min)
        return x.float()

    def invert(self, x):
        """
        Batch-wise inverse transform for x
        """
        x = x * (self.max - self.min) + self.min
        return x.float()
    
class ContTransformerClamp(nn.Module):
    """
    Transformer for Continuous Variables. Transforms to [0,1]
    """
    def __init__(self, x, min = None, max = None):
        super().__init__()
        self.min, self.max = min, max
        # if self.min is not None:
        #     self.min = torch.Tensor([min])
        # if self.max is not None:
        #     self.max = torch.Tensor([max])

    def forward(self, x):
        """
        Batch-wise transform for x
        """
        return x.float()

    def invert(self, x):
        """
        Batch-wise inverse transform for x
        """
        x = torch.clamp(x, self.min, self.max)
        return x.float()


class ContTransformerNegExp(nn.Module):
    """
    Neg-Exp Transformer for Continuous Variables. 
    With option to scale (= divide by .99 quantile)
    """
    def __init__(self, x, scale = True):
        super().__init__()
        self.max = torch.as_tensor(1.).to(torch.double)
        if scale:
            self.max = torch.max(x[~torch.isnan(x)]).to(torch.double)
        
        
    def forward(self, x):
        """
        Batch-wise transform for x: x -> exp(-x / scale)
        """
        x = x.to(torch.double) / self.max
        x = torch.exp(-x) - 1e-12
        return x.float()

    def invert(self, x):
        """
        Batch-wise inverse transform for x
        """
        x = - torch.log(x.to(torch.double) + 1e-12)
        x = x * self.max
        return x.float()


class ContTransformerLog(nn.Module):
    """
    Log-Transformer for Continuous Variables.

    logfun :: Can be torch.log
    expfun :: Can be torch.exp
    eps    :: Small number
    """
    def __init__(self, x, logfun = torch.log, expfun = torch.exp, eps = 1e-12):
        self.logfun = logfun
        self.expfun = expfun
        self.eps = eps
        super().__init__()
   
    def forward(self, x):
        """
        Batch-wise transform for x
        """
        x = self.logfun(x + self.eps)
        return x.float()

    def invert(self, x):
        """
        Batch-wise inverse transform for x
        """
        x = self.expfun(x) - self.eps
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


class ContTransformerInt(nn.Module):
    """

    Transform doubles to their nearest integer.
    Assumes that the input is integer. Therefore, the forward step is simply the identity.
    Note that the data type remains unchanged.

    """
    def __init__(self, x):
        super().__init__()

    def forward(self, x):
        """
        Identity.
        """
        return x.float()

    def invert(self, x):
        x = torch.round(x)
        return x.float()


class ContTransformerChain(nn.Module):
    """
    Chained transformer Continuous Variables. Chains several transforms.
    During forward pass, transforms are applied according to the list order,
    during invert, the order is reversed.
    """
    def __init__(self, x, tfms):  
        super().__init__()
        self.tfms = []
        for tf in tfms:
            itf = tf(x)
            x = itf.forward(x)
            self.tfms += [itf]
            
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


def tfms_chain(tfms):
    return partial(ContTransformerChain, tfms = tfms)

def _float_power(base, exp):
    """
    This is currently problematic due to numerical imprecision. torch.float_power would solve the problem 
    but currently can not be converted to ONNX.
    """
    out = torch.pow(base.to(torch.double), exp.to(torch.double))
    return out.to(torch.float64)

def float_pow10(base):
    return _float_power(base, torch.as_tensor(10.))


ContTransformerLog2Range = tfms_chain([partial(ContTransformerLog, logfun=torch.log2, expfun=torch.exp2), ContTransformerRange])
ContTransformerLogRange = tfms_chain([partial(ContTransformerLog, logfun=torch.log, expfun=torch.exp), ContTransformerRange])
ContTransformerNegExpRange = tfms_chain([partial(ContTransformerNegExp), ContTransformerRange])
ContTransformerClamp01Range = tfms_chain([partial(ContTransformerClamp, min=0., max = 1.), ContTransformerRange])
ContTransformerClamp0LogRange = tfms_chain([partial(ContTransformerClamp, min=0.), ContTransformerLog, ContTransformerRange])
