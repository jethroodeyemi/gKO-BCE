# transformer_arch.py

import math
import typing as ty
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor

# --- Activation Functions (Tabular GLUs) ---
def reglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)

def geglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)

def tanglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * torch.tanh(b)

def sigglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * torch.sigmoid(b)

def get_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
    if name == 'reglu':
        return reglu
    elif name == 'geglu':
        return geglu
    elif name == 'tanglu':
        return tanglu
    elif name == 'sigglu':
        return sigglu
    elif name == 'sigmoid':
        return torch.sigmoid
    else:
        return getattr(F, name)

# --- Feature/Hidden Mixup Data Augmentation ---
# Adapted from https://github.com/whatashot/excelformer
def batch_dim_shuffle(Xs: Tensor, beta: float = 0.5) -> ty.Tuple[Tensor, Tensor, Tensor]:
    """
    Performs dimension-wise mixup on hidden states for tabular transformers.
    """
    b, f, d = Xs.shape
    shuffle_rates = np.random.beta(beta, beta, size=(b, 1))
    dim_masks = np.random.random(size=(b, d)) < shuffle_rates  # Shape: (b, d)
    dim_masks = torch.from_numpy(dim_masks).to(Xs.device)

    shuffled_sample_ids = np.random.permutation(b)
    
    Xs_shuffled = Xs[shuffled_sample_ids]
    dim_masks = dim_masks.unsqueeze(1)  # Shape: (b, 1, d)
    Xs_mixup = dim_masks * Xs + ~dim_masks * Xs_shuffled

    return Xs_mixup, torch.from_numpy(shuffle_rates[:, 0]).float().to(Xs.device), torch.from_numpy(shuffled_sample_ids).to(Xs.device)

# ---  Weight Initializer ---
def attenuated_kaiming_uniform_(tensor, a=math.sqrt(5), scale=1., mode='fan_in', nonlinearity='leaky_relu'):
    """
    Kaiming uniform weight initializer scaled down by a factor to control gradient variance.
    """
    fan = nn_init._calculate_correct_fan(tensor, mode)
    gain = nn_init.calculate_gain(nonlinearity, a)
    std = gain * scale / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


class Tokenizer(nn.Module):
    """
    Transforms continuous numerical inputs into initial dense embeddings (tokens)
    for processing in the transformer attention layers.
    """
    def __init__(self, d_numerical: int, d_token: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(d_numerical, d_token))
        self.weight2 = nn.Parameter(Tensor(d_numerical, d_token))
        self.bias = nn.Parameter(Tensor(d_numerical, d_token)) if bias else None
        self.bias2 = nn.Parameter(Tensor(d_numerical, d_token)) if bias else None

        # Attenuated Initialization
        attenuated_kaiming_uniform_(self.weight)
        attenuated_kaiming_uniform_(self.weight2)
        if bias:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))
            nn_init.kaiming_uniform_(self.bias2, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight)

    def forward(self, x_num: Tensor) -> Tensor:
        # Input shape: (batch_size, n_features)
        # Output shape: (batch_size, n_features, d_token)
        x1 = self.weight[None] * x_num[:, :, None] + (self.bias[None] if self.bias is not None else 0.0)
        x2 = self.weight2[None] * x_num[:, :, None] + (self.bias2[None] if self.bias2 is not None else 0.0)
        return x1 * torch.tanh(x2)
    
class MultiheadAttention(nn.Module):
    """
    Standard PyTorch implementation of Multihead Attention with
    an attention mask and attenuated initialization of projection layers.
    """
    def __init__(self, d: int, n_heads: int, dropout: float, init_scale: float = 0.01) -> None:
        if n_heads > 1:
            assert d % n_heads == 0

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            attenuated_kaiming_uniform_(m.weight, scale=init_scale)
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            attenuated_kaiming_uniform_(self.W_out.weight)
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )
    
    def get_attention_mask(self, input_shape, device):
        # Causal mask over the feature-token axis. Returned as (1, seq_len, seq_len)
        # and broadcast over the (batch * n_heads) dimension so memory stays O(seq_len^2)
        # instead of O(batch * n_heads * seq_len^2). Values are identical either way.
        _, _, seq_len = input_shape
        seq_ids = torch.arange(seq_len, device=device)
        attention_mask = seq_ids[None, :] <= seq_ids[:, None]  # [i, j] True iff j <= i
        attention_mask = (1.0 - attention_mask.float()) * -1e4
        return attention_mask[None]  # (1, seq_len, seq_len)

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: ty.Optional[nn.Linear] = None,
        value_compression: ty.Optional[nn.Linear] = None,
    ) -> Tensor:
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)

        batch_size = len(q)
        # Compute the per-head key dim BEFORE reshaping (k.shape[-1] is still d here).
        # Reshaping first would make k.shape[-1] == d_head, and d_head // n_heads == 0,
        # producing sqrt(0) -> division by zero -> NaN.
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention_scores = q @ k.transpose(1, 2) / math.sqrt(d_head_key)
        
        masks = self.get_attention_mask(attention_scores.shape, attention_scores.device)
        attention = F.softmax(attention_scores + masks, dim=-1)
        
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


class EpitopeTransformer(nn.Module):
    """
    EpitopeTransformer: Tabular multi-head attention transformer classifier with
    GLU-based feedforward layers, attenuated weight initialization, and feature-wise
    hidden-mixup data augmentation.
    """
    def __init__(
        self,
        *,
        d_numerical: int,
        token_bias: bool,
        n_layers: int,
        d_token: int,
        n_heads: int,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        prenormalization: bool,
        kv_compression: ty.Optional[float] = None,
        kv_compression_sharing: ty.Optional[str] = None,
        d_out: int = 1,
        init_scale: float = 0.01,
        activation: str = 'tanglu',
    ) -> None:
        super().__init__()
        
        n_tokens = d_numerical
        self.tokenizer = Tokenizer(d_numerical, d_token, token_bias)

        def make_kv_compression():
            assert kv_compression
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression), bias=False
            )
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == 'layerwise'
            else None
        )

        def make_normalization():
            return nn.LayerNorm(d_token)
        
        self.activation_name = activation
        self.activation = get_activation_fn(activation)
        is_glu = activation.endswith('glu')

        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, init_scale=init_scale
                    ),
                    'linear0': nn.Linear(d_token, d_token * 2) if is_glu else nn.Linear(d_token, d_token),
                    'norm1': make_normalization(),
                }
            )
            attenuated_kaiming_uniform_(layer['linear0'].weight, scale=init_scale)
            nn_init.zeros_(layer['linear0'].bias)
            
            if not is_glu:
                layer['linear1'] = nn.Linear(d_token, d_token)
                attenuated_kaiming_uniform_(layer['linear1'].weight, scale=init_scale)
                nn_init.zeros_(layer['linear1'].bias)

            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.layers.append(layer)

        self.last_activation = nn.PReLU()
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout

        self.head = nn.Linear(d_token, d_out)
        attenuated_kaiming_uniform_(self.head.weight)
        
        self.last_fc = nn.Linear(n_tokens, 1)  # (b, n_tokens, d_token) -> (b, d_token)
        attenuated_kaiming_uniform_(self.last_fc.weight)

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
    
    def forward(self, x_num: Tensor, mixup: bool = False, beta: float = 0.5) -> ty.Union[Tensor, ty.Tuple[Tensor, Tensor, Tensor]]:
        x = self.tokenizer(x_num)
        
        if mixup:
            x, feat_masks, shuffled_ids = batch_dim_shuffle(x, beta=beta)

        for layer_idx, layer in enumerate(self.layers):
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                x_residual,
                x_residual,
                *self._get_kv_compressions(layer),
            )
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if 'linear1' in layer:
                x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        x = self.last_fc(x.transpose(1, 2))[:, :, 0]
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)

        if mixup:
            return x, feat_masks, shuffled_ids
        return x
