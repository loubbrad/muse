"""Includes (PyTorch) transformer model and config classes."""

import torch
from torch import nn as nn
from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model: int = 512
    n_heads: int = 16
    n_layers: int = 12
    vocab_size: int = -1  # Set according to tokenizer
    max_seq_len: int = 1024


class Transformer(nn.Module):
    def __init__(
        self,
    ):
        raise NotImplementedError

    def forward(
        self,
    ):
        raise NotImplementedError
