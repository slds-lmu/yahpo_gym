import typing as ty

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init

from fastai.tabular.all import *

from yahpo_train.cont_scalers import *
from yahpo_train.models_utils import *

class AbstractSurrogate(nn.Module):

    def __init__(self):
        super().__init__()
        
    def _build_embeddings(self, dls, embds_dbl=None, embds_tgt=None, emb_szs=None):
        self._build_embeddings_xcat(dls=dls, emb_szs=emb_szs)
        self._build_embeddings_xcont(dls=dls, embds_dbl=embds_dbl)
        self._build_embeddings_y(dls=dls, embds_tgt=embds_tgt)
        self.n_inputs = self.n_emb + self.n_cont

    def _build_embeddings_xcat(self, dls, emb_szs):
        # Categorical Embeddings
        emb_szs = get_emb_sz(dls.train_ds, {} if emb_szs is None else emb_szs)
        self.embds_fct = nn.ModuleList([Embedding(ni, nf) for ni, nf in emb_szs])
        # Init with Kaiming
        [nn_init.kaiming_uniform_(embd.weight, a=math.sqrt(5)) for embd in self.embds_fct]
        self.n_emb = sum(e.embedding_dim for e in self.embds_fct)

    def _build_embeddings_xcont(self, dls, embds_dbl):
        # Transform continuous variables and targets
        if embds_dbl is not None:
            self.embds_dbl = nn.ModuleList([f(torch.from_numpy(cont[1].values).float()) for cont, f in zip(dls.all_cols[dls.cont_names].iteritems(), embds_dbl)])
        else:
            self.embds_dbl = nn.ModuleList([ContTransformerRange(torch.from_numpy(cont.values).float()) for _, cont in dls.all_cols[dls.cont_names].iteritems()])
        self.n_cont = len(dls.cont_names)
    
    def _build_embeddings_y(self, dls, embds_tgt=None):
        if embds_tgt is not None:
            self.embds_tgt = nn.ModuleList([f(torch.from_numpy(cont[1].values).float()) for cont, f in zip(dls.ys[dls.y_names].iteritems(), embds_tgt)])
        else:
            self.embds_tgt = nn.ModuleList([ContTransformerRange(torch.from_numpy(cont.values).float()) for name, cont in dls.ys[dls.y_names].iteritems()])

    def _embed_features(self, x_cat, x_cont):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embds_fct)]
            x = torch.cat(x, 1)
        if self.n_cont != 0:
            xd = [e(x_cont[:,i]).unsqueeze(1) for i,e in enumerate(self.embds_dbl)]
            xd = torch.cat(xd, 1)
            x = torch.cat([x, xd], 1) if self.n_emb > 0 else xd    
        return x

    def trafo_ys(self, ys):
        ys = [e(ys[:,i]).unsqueeze(1) for i,e in enumerate(self.embds_tgt)]
        ys = torch.cat(ys, 1)
        return ys

    def inv_trafo_ys(self, ys):
        ys = [e.invert(ys[:,i]).unsqueeze(1) for i,e in enumerate(self.embds_tgt)]
        ys = torch.cat(ys, 1)
        return ys
    
    def forward(self):
        raise NotImplementedError

    def export_onnx(self, config_dict, device='cuda:0'):
        """
        Export model to an ONNX file. We can safely ignore tracing errors with respect to lambda since lambda will be constant during inference.
        """
        self.eval()
        torch.onnx.export(self,
            (torch.ones(1, len(config_dict.cat_names), dtype=torch.int, device=device), {'x_cont': torch.randn(1, len(config_dict.cont_names), device=device)}),
            config_dict.get_path("model"),
            do_constant_folding=True,
            export_params=True,
            input_names=['x_cat', 'x_cont'],
            opset_version=12
        )



class FFSurrogateModel(AbstractSurrogate):
    def __init__(self, dls, layers = [400, 400], deeper = [400, 400, 400], wide = True, use_bn = False, ps=0.1, act_cls=nn.SELU(inplace=True), final_act = nn.Sigmoid(), lin_first=False, embds_dbl=None, embds_tgt=None, emb_szs=None) -> None:
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
            



### ResNet
class ResNet(AbstractSurrogate):
    def __init__(self, dls, embds_dbl=None, embds_tgt=None, emb_szs=None, d: int = 256, d_hidden_factor: float =2,
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
                        'norm': (self._make_normalization(normalization)(d)),
                        'linear0': nn.Linear(
                            d, d_hidden * (2 if activation.endswith('glu') else 1)
                        ),
                        'linear1': nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.last_normalization = self._make_normalization(normalization)(d)
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

    def _make_normalization(self, normalization:str):
        if normalization == 'batchnorm':
            normalization = nn.BatchNorm1d
        else:
            normalization = nn.LayerNorm
        return normalization



class Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
    ) -> None:
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_cat: ty.Optional[Tensor], x_cont: Tensor) -> Tensor:
        x_some = x_cont if x_cat is None else x_cat
        assert x_some is not None
        x_cont = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_cont is None else [x_cont]),
            dim=1,
        )
        x = self.weight[None] * x_cont[:, :, None]
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]
        return x


class MultiheadAttention(nn.Module):
    def __init__(
        self, d: int, n_heads: int, dropout: float, initialization: str
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: ty.Optional[nn.Linear],
        value_compression: ty.Optional[nn.Linear],
    ) -> Tensor:
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x


class Transformer(AbstractSurrogate):
    """Transformer.
    Repurposed and adapted from https://github.com/yandex-research/rtdl under Apache License 2.0
    References:
    - https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
    - https://github.com/facebookresearch/pytext/tree/master/pytext/models/representations/transformer
    - https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/linformer/linformer_src/modules/multihead_linear_attention.py#L19
    """

    def __init__(
        self,
        dls,
        embds_dbl:typing.List=None,
        embds_tgt:typing.List=None,
        d_token:int = 192,
        final_act = nn.Sigmoid(),
        # Tokenizer
        token_bias: bool = True,
        # transformer
        n_layers: int = 3,

        n_heads: int = 8,
        d_ffn_factor: float = 4/3,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        activation: str = 'reglu',
        prenormalization: bool = True,
        initialization: str = 'kaiming',

        # linformer
        kv_compression: ty.Optional[float] = None,
        kv_compression_sharing: ty.Optional[str] = None,
    ) -> None:
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)

        super().__init__()

        self._build_embeddings_xcont(dls, embds_dbl)
        self.tokenizer = Tokenizer(len(dls.cont_names), [len(dls.train.classes[n]) for n in dls.train.cat_names], d_token, token_bias)
        n_tokens = self.tokenizer.n_tokens

        self.final_act = final_act
        self._build_embeddings_y(dls, embds_tgt)

        def make_kv_compression():
            assert kv_compression
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression), bias=False
            )
            if initialization == 'xavier':
                nn_init.xavier_uniform_(compression.weight)
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == 'layerwise'
            else None
        )

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden * (2 if activation.endswith('glu') else 1)
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.layers.append(layer)

        self.activation = get_activation_fn(activation)
        self.last_activation = get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, dls.ys.shape[1])

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x_cat: ty.Optional[Tensor], x_cont: Tensor =None, invert_ytrafo: bool = True) -> Tensor:
        
        # Transform continuous features
        if self.n_cont != 0:
            xcont = [e(x_cont[:,i]).unsqueeze(1) for i,e in enumerate(self.embds_dbl)]
            xcont = torch.cat(xcont, 1)

        x = self.tokenizer(x_cat, x_cont)

        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                # for the last attention, it is enough to process only [CLS]
                (x_residual[:, :1] if is_last_layer else x_residual),
                x_residual,
                *self._get_kv_compressions(layer),
            )
            if is_last_layer:
                x = x[:, : x_residual.shape[1]]
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        assert x.shape[1] == 1
        x = x[:, 0]
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        
        y = self.final_act(x)
        if invert_ytrafo:
            return self.inv_trafo_ys(y)
        else:
            return y

