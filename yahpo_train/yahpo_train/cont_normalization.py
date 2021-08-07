import torch
import torch.nn as nn
from scipy import optimize

class ContNormalization(nn.Module):
    """
    Yeoh-Johnson Transformer
    Learns the transformation during initialization and
    outputs the transformation afterwards.
    """
    def __init__(self, x_sample, lmbda = None, eps=1e-6, normalize='scale', sigmoid_p = .03):
        super(ContNormalization, self).__init__()
        self.eps = eps
        self.normalize, self.sigmoid_p = normalize, to_tensor(sigmoid_p)
        if not lmbda:
            self.lmbda  = self.est_params(to_tensor(x_sample))
        else:
            self.lmbda = to_tensor(lmbda)
        xt = self.trafo_yj(x_sample, self.lmbda)

        if self.normalize == 'scale':
            self.sigma, self.mu = torch.var_mean(xt)
        elif self.normalize == 'range':
            self.min, self.max = torch.min(xt), torch.max(xt)
            
    def forward(self, x):
        x = self.trafo_yj(x.float(), self.lmbda)
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
        x = self.inverse_trafo_yj(x, self.lmbda) 
        return x

    def trafo_yj(self, x, lmbda):   
        return torch.where(x >= 0, self.scale_pos(x, lmbda), self.scale_neg(x, lmbda))

    def scale_pos(self, x, lmbda):
        if torch.abs(lmbda) < self.eps:
            x = torch.log1p(x)
        else:
            x = (torch.float_power(x + 1., lmbda) - 1) / lmbda 
        return x
    def scale_neg(self, x, lmbda):
        if torch.abs(lmbda - 2.) <= self.eps:
            x = torch.log1p(-x)
        else:
            x = (torch.float_power(-x + 1, 2. - lmbda) - 1.) / (2. - lmbda)
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
        res = optimize.minimize_scalar(lambda lmbd: self._neg_loglik(lmbd, x_sample), bounds=(-10, 10), method='bounded')
        return to_tensor(res.x)

    def inverse_trafo_yj(self, x, lmbda):
        return torch.where(x >= 0, self.inv_pos(x, lmbda), self.inv_neg(x, lmbda))

    def inv_pos(self, x, lmbda):
        if torch.abs(lmbda) < self.eps:
            x = torch.expm1(x)
        else:
            x = torch.float_power(x * lmbda + 1, 1 / lmbda) - 1.
        return x
    
    def inv_neg(self, x,  lmbda):
        if torch.abs(lmbda - 2.) < self.eps:
            x = -torch.exp(x)+1
        else:
            x = 1 - torch.float_power(-(2.-lmbda) * x + 1. , 1. / (2. - lmbda))
        return x

def to_tensor(x):
    if torch.is_tensor(x):
        return x
    else:
        return torch.as_tensor(x, dtype=torch.float64)
