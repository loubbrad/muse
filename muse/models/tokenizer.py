"""Includes tokenizer for converting piano-roll to PyTorch tensors for
pre-training, fine-tuning, and inference."""

import torch


from models.model import ModelConfig
from pianoroll import PianoRoll


# TODO: Find out how to properly implement this abstract class.
class Tokenizer:
    """Abstract class for Tokenizers."""

    def __init__(self, model_config: ModelConfig, device: str, mode: str):
        pass

    def encode(self):
        pass

    def decode(self):
        pass

    def _seq(self):
        pass


class PretrainTokenizer(Tokenizer):
    """Tokenizes and sequentialises PianoRoll objects for pre-training.

    Args:
        model_config (model.ModelConfig): Config for model.
        device (str): Device to sent PyTorch tensors to send torch.tensors to.
    """

    def __init__(self, model_config: ModelConfig, device: str):
        super().__init__()  # Needed?
        raise NotImplementedError

    def encode(self):
        raise NotImplementedError

    def decode(self):
        raise NotImplementedError

    def _seq(self):
        raise NotImplementedError


class FinetuneTokenizer(Tokenizer):
    """Tokenizes and sequentialises PianoRoll objects for pre-training.

    Args:
        model_config (model.ModelConfig): Config for model.
        device (str): Device to sent PyTorch tensors to send torch.tensors to.
    """

    def __init__(self, model_config: ModelConfig, device: str):
        super().__init__()  # Needed?
        raise NotImplementedError

    def encode(self):
        raise NotImplementedError

    def decode(self):
        raise NotImplementedError

    def _seq(self):
        raise NotImplementedError
