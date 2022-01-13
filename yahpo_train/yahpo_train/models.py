import torch
import torch.nn as nn
import torch.onnx
import torch.nn.functional as F
from fastai.tabular.all import *
import typing as ty

from yahpo_train.embed_helpers import *
from yahpo_train.cont_scalers import *


def _export_onnx(elf, config_dict, device='cuda:0'):
    self.eval()
    torch.onnx.export(self,
        (torch.ones(1, len(config_dict.cat_names), dtype=torch.int, device=device), {'x_cont': torch.randn(1, len(config_dict.cont_names), device=device)}),
        config_dict.get_path("model"),
        do_constant_folding=True,
        export_params=True,
        input_names=['x_cat', 'x_cont'],
        opset_version=12
    )


class AbstractSurrogate(nn.Module):

    def __init__(self):
        super().__init__()
        
    def _build_embeddings(self, dls, embds_dbl=None, embds_tgt=None, emb_szs=None):
        emb_szs = get_emb_sz(dls.train_ds, {} if emb_szs is None else emb_szs)
        self.embds_fct = nn.ModuleList([Embedding(ni, nf) for ni, nf in emb_szs])

        # Transform continuous variables and targets
        if embds_dbl is not None:
            self.embds_dbl = nn.ModuleList([f(torch.from_numpy(cont[1].values).float()) for cont, f in zip(dls.all_cols[dls.cont_names].iteritems(), embds_dbl)])
        else:
            self.embds_dbl = nn.ModuleList([ContTransformerRange(torch.from_numpy(cont.values).float()) for name, cont in dls.all_cols[dls.cont_names].iteritems()])
       
        if embds_tgt is not None:
            self.embds_tgt = nn.ModuleList([f(torch.from_numpy(cont[1].values).float()) for cont, f in zip(dls.ys[dls.y_names].iteritems(), embds_tgt)])
        else:
            self.embds_tgt = nn.ModuleList([ContTransformerRange(torch.from_numpy(cont.values).float()) for name, cont in dls.ys[dls.y_names].iteritems()])

        self.n_emb,self.n_cont = sum(e.embedding_dim for e in self.embds_fct), len(dls.cont_names)

        self.n_inputs = [self.n_emb + self.n_cont]

    def _embed_features(self, x_cat, x_cont):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embds_fct)]
            x = torch.cat(x, 1)
        if self.n_cont != 0:
            xd = [e(x_cont[:,i]).unsqueeze(1) for i,e in enumerate(self.embds_dbl)]
            xd = torch.cat(xd, 1)
            x = torch.cat([x, xd], 1) if self.n_emb > 0 else xd

    def trafo_ys(self, ys):
        ys = [e(ys[:,i]).unsqueeze(1) for i,e in enumerate(self.embds_tgt)]
        ys = torch.cat(ys, 1)
        return ys

    def inv_trafo_ys(self, ys):
        ys = [e.invert(ys[:,i]).unsqueeze(1) for i,e in enumerate(self.embds_tgt)]
        ys = torch.cat(ys, 1)
        return ys
    

    def forward():
        raise NotImplementedError
        


class FFSurrogateModel(AbstractSurrogate):
    def __init__(self, dls, layers = [400, 400], deeper = [400, 400, 400], wide = True, use_bn = False, ps=0.1, act_cls=nn.SELU(inplace=True), final_act = nn.Sigmoid(), lin_first=False, embds_dbl=None, embds_tgt=None, emb_szs = None) -> None:
        super().__init__()
        self._build_embeddings(dls=dls, embds_dbl=embds_dbl, embds_tgt=embds_tgt, emb_szs=emb_szs)

        if not (len(layers) | len(deeper) | wide):
            raise Exception("One of layers, deeper or wide has to be set!")

        self.sizes = [self.n_inputs] + layers + [dls.ys.shape[1]]
        self.deep, self.deeper, self.wide = nn.Sequential(), nn.Sequential(), nn.Sequential()
    
        # Deep Part
        if len(layers):
            ps1 = [ps for i in layers]
            self.sizes = [self.n_emb + self.n_cont] + layers + [dls.ys.shape[1]]
            actns = [act_cls for _ in range(len(self.sizes)-2)] + [None]
            _layers_deep = [LinBnDrop(self.sizes[i], self.sizes[i+1], bn=use_bn and i!=len(actns)-1, p=p, act=a, lin_first=lin_first)
                        for i,(p,a) in enumerate(zip(ps1+[0.],actns))]
            self.deep = nn.Sequential(*_layers_deep)

        # Deeper part
        if len(deeper):
            ps2 = [ps for i in deeper]
            self.deeper_sizes = [self.n_emb + self.n_cont] + deeper + [dls.ys.shape[1]]
            deeper_actns = [act_cls for _ in range(len(deeper))] + [None]
            _layers_deeper = [LinBnDrop(self.deeper_sizes[i], self.deeper_sizes[i+1], bn=use_bn and i!=len(deeper_actns)-1, p=p, act=a, lin_first=lin_first) for i,(p,a) in enumerate(zip(ps2+[0.],deeper_actns))]
            self.deeper = nn.Sequential(*_layers_deeper)

        if wide:
            self.wide = nn.Sequential(nn.Linear(self.sizes[0], self.sizes[-1]))

        self.final_act = final_act
        
    def forward(self, x_cat, x_cont=None, invert_ytrafo=True):

        x = self._embed_features(x_cat, x_cont)

        xs = torch.zeros(x.shape[0], self.sizes[-1], device = x.device)
        if len(self.wide):
            xs = xs.add(self.wide(x))
        if len(self.deep):
            xs = xs.add(self.deep(x))
        if len(self.deeper):
            xs = xs.add(self.deeper(x))

        y = self.final_act(xs)
        if invert_ytrafo:
            return self.inv_trafo_ys(y)
        else:
            return y
            

    def export_onnx(self, config_dict, device='cuda:0'):
        """
        Export model to an ONNX file. We can safely ignore tracing errors with respect to lambda since lambda will be constant during inference.
        """
        _export_onnx(self, config_dict, device='cuda:0')


### ResNet

class ResNet(AbstractSurrogate):
    def __init__(self, dls, embds_dbl, embds_tgt, emb_szs, d: int = 256, d_hidden_factor: float =2,
        n_layers: int = 4, activation: str = "reglu", normalization: str ="batchnorm", hidden_dropout: float = .0, residual_dropout: float = .2, final_act = nn.Sigmoid(),) -> None:
        """
        ResNet model.
        Repurposed and adapted from https://github.com/yandex-research/rtdl under Apache License 2.0
        dls :: DatasetLoader
        embds_dbl :: Numeric Embeddings
        embds_tgt :: Target Embeddings
        d :: dimensionality of the hidden space
        d_hidden_factor :: factor by which the hidden dimension is reduced
        n_layers :: number of layers
        activation :: activation function
        normalization :: normalization function
        hidden_dropout :: dropout rate for hidden layers
        residual_dropout :: dropout rate for residual connections
        """
        super().__init__()
        self._build_embeddings(dls=dls, embds_dbl=embds_dbl, embds_tgt=embds_tgt, emb_szs=emb_szs)
    
        self.main_activation = get_activation_fn(activation)
        self.last_activation = get_nonglu_activation_fn(activation)
        self.residual_dropout = residual_dropout
        self.hidden_dropout = hidden_dropout
        self.final_act = final_act

        d_hidden = int(d * d_hidden_factor)
        self.first_layer = nn.Linear(self.n_inputs, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        'norm': (self._get_normalization(normalization)(d)),
                        'linear0': nn.Linear(
                            d, d_hidden * (2 if activation.endswith('glu') else 1)
                        ),
                        'linear1': nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.last_normalization = self._get_normalization(normalization)
        self.head = nn.Linear(d, dls.ys.shape[1])

    def forward(self, x_cat, x_cont=None, invert_ytrafo=True) -> Tensor:
        x = self._embed_features(x_cat, x_cont)
        x = self.first_layer(x)
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            z = x
            z = layer['norm'](z)
            z = layer['linear0'](z)
            z = self.main_activation(z)
            if self.hidden_dropout:
                z = F.dropout(z, self.hidden_dropout, self.training)
            z = layer['linear1'](z)
            if self.residual_dropout:
                z = F.dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        y = self.final_act(x)
        if invert_ytrafo:
            return self.inv_trafo_ys(y)
        else:
            return y
        return y

    def _make_normalization(self, normalization:str):
        if normalization == 'batchnorm':
            self.normalization = nn.BatchNorm1d
        else:
            self.normalization = nn.LayerNorm


def reglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class ReGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)

def get_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
    return (
        reglu
        if name == 'reglu'
        else geglu
        if name == 'geglu'
        else torch.sigmoid
        if name == 'sigmoid'
        else getattr(F, name)
    )

def get_nonglu_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
    return (
        F.relu
        if name == 'reglu'
        else F.gelu
        if name == 'geglu'
        else get_activation_fn(name)
    )

