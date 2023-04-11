"""Includes (PyTorch) transformer model and config classes."""

import math
import torch
import torch.utils.checkpoint
from torch import nn as nn
from torch.nn import functional as F
from typing import Tuple
from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model: int = 528
    n_heads: int = 22
    n_layers: int = 8
    ff_mult: int = 4
    drop_p = 0.0
    max_seq_len: int = 320

    # Set according to tokenizer
    vocab_size: int = -1
    pad_id: int = -1
    mask_id: int = -1

    grad_checkpoint: bool = False
    att_mask: bool = None


# Taken from facebookresearch/llama/model.py
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


# Taken from facebookresearch/llama/model.py
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class ManualEncoderBlock(nn.Module):
    """Manually implemented transformer encoder block.

    This block has the following changes from a typical transformer encoder:

        - Rotary embeddings are applied to the key/query matrices.
        - Layer norm is applied before attention and feed forward, instead of
            after.
        - Keys arising from padding are masked during attention.
        - GELU activation is used instead of ReLU.

    Args:
        model_config (ModelConfig): Model config settings.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.pad_id = model_config.pad_id
        self.n_heads = model_config.n_heads
        self.d_head = model_config.d_model // model_config.n_heads
        self.max_seq_len = model_config.max_seq_len

        # Attention
        self.q = nn.Linear(
            in_features=model_config.d_model,
            out_features=model_config.d_model,
            bias=False,
        )
        self.k = nn.Linear(
            in_features=model_config.d_model,
            out_features=model_config.d_model,
            bias=False,
        )
        self.v = nn.Linear(
            in_features=model_config.d_model,
            out_features=model_config.d_model,
            bias=False,
        )
        self.att_proj_linear = nn.Linear(
            in_features=model_config.d_model,
            out_features=model_config.d_model,
        )
        self.att_dropout = nn.Dropout(model_config.drop_p)
        self.resid_dropout = nn.Dropout(model_config.drop_p)

        # FF Layer
        self.ff_dropout = nn.Dropout(model_config.drop_p)
        self.ff_linear_1 = nn.Linear(
            in_features=model_config.d_model,
            out_features=model_config.d_model * model_config.ff_mult,
        )
        self.ff_linear_2 = nn.Linear(
            in_features=model_config.d_model * model_config.ff_mult,
            out_features=model_config.d_model,
        )
        self.ff_activation = nn.GELU()

        # Pre layer norms
        self.norm1 = nn.LayerNorm(model_config.d_model)
        self.norm2 = nn.LayerNorm(model_config.d_model)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        x = x + self._att_block(self.norm1(x), freqs_cis)
        x = x + self._ff_block(self.norm2(x))

        return x

    def _att_block(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.q(x), self.k(x), self.v(x)

        # Reshape for rotary embeddings
        xq = xq.view(batch_size, seq_len, self.n_heads, self.d_head)
        xk = xk.view(batch_size, seq_len, self.n_heads, self.d_head)
        xv = xv.view(batch_size, seq_len, self.n_heads, self.d_head)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Reshape for attention calculation: (b_sz, n_head, s_len, d_head)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Calc attention matrix:
        # (b_sz, n_head, s_len, d_head) @ (b_sz, n_head, d_head, s_len)
        #  = (b_sz, n_head, s_len, s_len)
        att = (xq @ xk.transpose(2, 3)) / math.sqrt(self.d_head)
        att = F.softmax(att, dim=-1)
        att = self.att_dropout(att)

        # Calc outputs:
        # (b_sz, n_head, s_len, s_len) @ (b_sz, n_head, s_len, d_head)
        # = (b_sz, n_head, s_len, d_head)
        out = att @ xv
        out = out.transpose(1, 2).contiguous()  # (b_sz, s_len, n_head, d_head)
        out = out.view(batch_size, seq_len, self.n_heads * self.d_head)

        return self.resid_dropout(self.att_proj_linear(out))

    def _ff_block(self, x: torch.Tensor):
        x = self.ff_linear_2(self.ff_activation(self.ff_linear_1(x)))

        return self.ff_dropout(x)


class FusedEncoderBlock(nn.Module):
    """Transformer encoder block using F.scaled_dot_product_attention().

    This block has the following changes from a typical transformer encoder:

        - Rotary embeddings are applied to the key/query matrices.
        - Layer norm is applied before attention and feed forward, instead of
            after.
        - Keys arising from padding are masked during attention.
        - GELU activation is used instead of ReLU.

    Args:
        model_config (ModelConfig): Model config settings.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.pad_id = model_config.pad_id
        self.drop_p = model_config.drop_p
        self.n_heads = model_config.n_heads
        self.d_head = model_config.d_model // model_config.n_heads
        self.max_seq_len = model_config.max_seq_len

        # Attention
        self.q = nn.Linear(
            in_features=model_config.d_model,
            out_features=model_config.d_model,
            bias=False,
        )
        self.k = nn.Linear(
            in_features=model_config.d_model,
            out_features=model_config.d_model,
            bias=False,
        )
        self.v = nn.Linear(
            in_features=model_config.d_model,
            out_features=model_config.d_model,
            bias=False,
        )
        self.att_proj_linear = nn.Linear(
            in_features=model_config.d_model,
            out_features=model_config.d_model,
        )
        self.resid_dropout = nn.Dropout(model_config.drop_p)

        # FF Layer
        self.ff_dropout = nn.Dropout(model_config.drop_p)
        self.ff_linear_1 = nn.Linear(
            in_features=model_config.d_model,
            out_features=model_config.d_model * model_config.ff_mult,
        )
        self.ff_linear_2 = nn.Linear(
            in_features=model_config.d_model * model_config.ff_mult,
            out_features=model_config.d_model,
        )
        self.ff_activation = nn.GELU()

        # Pre layer norms
        self.norm1 = nn.LayerNorm(model_config.d_model)
        self.norm2 = nn.LayerNorm(model_config.d_model)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        x = x + self._att_block(self.norm1(x), freqs_cis)
        x = x + self._ff_block(self.norm2(x))

        return x

    def _att_block(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.q(x), self.k(x), self.v(x)

        # Reshape for rotary embeddings
        xq = xq.view(batch_size, seq_len, self.n_heads, self.d_head)
        xk = xk.view(batch_size, seq_len, self.n_heads, self.d_head)
        xv = xv.view(batch_size, seq_len, self.n_heads, self.d_head)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Reshape for attention calculation: (b_sz, n_head, s_len, d_head)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Required as we are not using a nn.Dropout layer
        if self.training:
            att_dropout = self.drop_p
        else:
            att_dropout = 0.0

        # Using beta torch functionality (subject to change)
        # See - https://shorturl.at/jtI17
        att = F.scaled_dot_product_attention(
            query=xq,
            key=xk,
            value=xv,
            dropout_p=att_dropout,
            is_causal=False,
        )

        # Shape (b_sz, s_len, n_head, d_head)
        out = att.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.n_heads * self.d_head)

        return self.resid_dropout(self.att_proj_linear(out))

    def _ff_block(self, x: torch.Tensor):
        x = self.ff_linear_2(self.ff_activation(self.ff_linear_1(x)))

        return self.ff_dropout(x)


class MuseEncoder(nn.Module):
    """MuseEncoder with no additional model head.

    Args:
        model_config (ModelConfig): Model config settings.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.model_config = model_config
        self.pad_id = model_config.pad_id

        # Used for Rotary Embeddings - see LLaMA
        d_head = model_config.d_model // model_config.n_heads
        max_seq_len = model_config.max_seq_len
        self.register_buffer(
            "freqs_cis",
            self._precompute_freqs_cis(d_head, max_seq_len),
        )

        self.tok_embeddings = nn.Embedding(
            num_embeddings=model_config.vocab_size,
            embedding_dim=model_config.d_model,
            padding_idx=model_config.pad_id,
        )

        self.encode_layers = nn.ModuleList()
        for _ in range(model_config.n_layers):
            self.encode_layers.append(FusedEncoderBlock(model_config))

    def forward(self, src: torch.Tensor):
        """Forward pass of MuseEncoder.

        Args:
            src (torch.tensor): Input to encoder block, of shape (batch_size,
                seq_len, d_model).

        Returns:
            torch.tensor: Model outputs with shape (batch_size, seq_len,
                d_model).
        """
        hidden_states = self.tok_embeddings(src)

        # Implements gradient checkpoints on Encoder Layers.
        if self.model_config.grad_checkpoint is True:
            for layer in self.encode_layers:

                def create_custom_forward(module):
                    def custom_forward(*args):
                        return module(*args)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    self.freqs_cis,
                    preserve_rng_state=True,
                )

        else:
            for layer in self.encode_layers:
                hidden_states = layer(hidden_states, self.freqs_cis)

        return hidden_states

    # Taken from facebookresearch/llama/model.py
    def _precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0):
        freqs = 1.0 / (
            theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
        )
        t = torch.arange(end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

        return freqs_cis


class ChoraleEncoder(nn.Module):
    """Old counterpoint encoder, for testing purposes."""

    def __init__(self, config: ModelConfig):
        super(ChoraleEncoder, self).__init__()
        self.seq_len = config.max_seq_len
        self.vocab_len = config.vocab_size
        self.emb_dim = config.d_model
        self.ff_dim = config.d_model * config.ff_mult
        self.pad_idx = config.pad_id
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers

        # Input layer
        self.pos_emb = nn.Embedding(self.seq_len, self.emb_dim)
        self.key_embed = nn.Embedding(
            self.vocab_len, self.emb_dim, padding_idx=self.pad_idx
        )  # pad_idx required?

        # Transformer
        self.encoder_layer = nn.TransformerEncoderLayer(
            self.emb_dim,
            self.n_heads,
            self.ff_dim,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, self.n_layers)

    def forward(self, seq: torch.tensor):
        # Embed and encode
        pos = torch.arange(0, self.seq_len, dtype=torch.long).to(seq.device)
        pad_mask = seq == self.pad_idx  # Shape (#batches, seq_len)
        emb = self.key_embed(seq) + self.pos_emb(
            pos
        )  # Shape (#batches, seq_len, emb_dim) + (1, seq_len, emb_dim)
        enc = self.encoder(
            src=emb, src_key_padding_mask=pad_mask
        )  # Shape (#batches, seq_len, emb_dim)

        return enc


class MuseMaskedLM(nn.Module):
    """MuseEncoder with head for masked language modelling.

    Args:
        model_config (ModelConfig): Model config settings.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.model = MuseEncoder(model_config)
        self.lm_head = nn.Linear(
            model_config.d_model, model_config.vocab_size, bias=False
        )

    def forward(self, src: torch.Tensor):
        """Forward pass of MuseEncoder with MaskedLM head (logits output).

        Args:
            src (torch.tensor): Input to encoder block, of shape (batch_size,
                seq_len, d_model).

        Returns:
            torch.tensor: Forward pass of src through the encoder block.
        """
        logits = self.lm_head(self.model(src))

        return logits


def main():
    conf = ModelConfig()
    conf.vocab_size = 150
    conf.pad_id = 2
    model = MuseMaskedLM(conf).cuda()

    x = (
        (torch.rand((2, conf.max_seq_len)) * conf.vocab_size)
        .abs()
        .to(torch.int)
    ).cuda()
    y = model.forward(x)

    print(x)
    print(y)
    print(x.shape)
    print(y.shape)


if __name__ == "__main__":
    main()
