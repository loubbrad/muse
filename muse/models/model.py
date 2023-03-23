"""Includes (PyTorch) transformer model and config classes."""

import torch
from torch import nn as nn
from dataclasses import dataclass


"""Notes:

Use xformers library for attention and rotary embeddings.


"""


@dataclass
class ModelConfig:
    d_model: int = 528
    n_heads: int = 22
    n_layers: int = 16
    max_seq_len: int = 1024

    # Set according to tokenizer
    vocab_size: int = -1
    pad_id: int = -1
    mask_id: int = -1


class Transformer(nn.Module):
    def __init__(
        self,
    ):
        raise NotImplementedError

    def forward(
        self,
    ):
        raise NotImplementedError
