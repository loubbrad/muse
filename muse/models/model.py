"""Includes (PyTorch) transformer model and config classes. Created using
the xFormers library."""

import torch
from torch import nn as nn
import torch.utils.checkpoint
from xformers.factory import xFormerEncoderBlock, xFormerEncoderConfig
from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model: int = 384
    n_heads: int = 16
    n_layers: int = 2
    ff_mult: int = 4
    drop_p = 0.1
    max_seq_len: int = 1024

    # Set according to tokenizer
    vocab_size: int = -1
    pad_id: int = -1
    mask_id: int = -1

    grad_checkpoint: bool = True
    att_mask: bool = None


class EncoderBlock(nn.Module):
    """Encoder block with rotary embeddings from xFormers library.

    Note that xFormer blocks expect batch first.

    Args:
        model_config (ModelConfig): Model config settings.
    """

    def __init__(self, model_config: ModelConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.mask = model_config.att_mask

        encoder_config = {
            "dim_model": model_config.d_model,
            "residual_norm_style": "pre",
            "multi_head_config": {
                "num_heads": model_config.n_heads,
                "residual_dropout": model_config.drop_p,
                "use_rotary_embeddings": True,
                "attention": {
                    "name": "scaled_dot_product",
                    "dropout": model_config.drop_p,
                    "seq_len": model_config.max_seq_len,
                    "casual": False,
                    "use_rotary_embeddings": True,
                },
            },
            "feedforward_config": {
                "name": "MLP",
                "dropout": model_config.drop_p,
                "activation": "gelu",
                "hidden_layer_multiplier": 4,
            },
        }

        config = xFormerEncoderConfig(**encoder_config)
        self.encoder = xFormerEncoderBlock(config)

    def forward(self, src: torch.Tensor):
        """Forward pass for EncoderBlock.

        Args:
            src (torch.tensor): Input to encoder block, of shape (batch_size,
                seq_len, d_model).

        Returns:
            torch.tensor: forward pass of src through the encoder block.
        """
        return self.encoder(src, self.mask)


class MuseEncoder(nn.Module):
    """MuseEncoder with no additional model head.

    Args:
        model_config (ModelConfig): Model config settings.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.model_config = model_config

        self.tok_embeddings = nn.Embedding(
            num_embeddings=model_config.vocab_size,
            embedding_dim=model_config.d_model,
            padding_idx=model_config.pad_id,
        )

        self.encode_layers = nn.ModuleList()
        for layer_id in range(model_config.n_layers):
            self.encode_layers.append(EncoderBlock(model_config, layer_id))

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
        # TODO: Test that this doesn't change the gradient calculation
        # TODO: Do profiling for the memory/compute tradeoff
        if self.model_config.grad_checkpoint is True:
            for layer in self.encode_layers:

                def create_custom_forward(module):
                    def custom_forward(hidden_states):
                        return module(hidden_states)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    preserve_rng_state=True,
                )

        else:
            for layer in self.encode_layers:
                hidden_states = layer(hidden_states)

        return hidden_states


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
    pass


if __name__ == "__main__":
    main()
