import json

from .pianoroll import PianoRoll
from .datasets import Dataset


# TODO: Implement
class Tokenizer:
    """Tokenizes PianoRoll objects to be fed into PyTorch model

    Args:
        dataset (datasets.Dataset): Piano-roll dataset to be tokenized.
    """

    def __init__(
        self,
        dataset: Dataset,
        model_config,  # Add type hint when implemented
        **kwargs,  # State explicitly when implemented
    ):
        """Initialise tokenizer according to dataset and config."""

        raise (NotImplementedError)

    def save(self, save_path: str):
        """Saves Tokenizer configuration for reloading.

        Args:
            save_path (str):
        """

    @classmethod
    def from_preset(path: str | None, name: str | None):
        """

        Args:
            path (str | None): _description_
            name (str | None): _description_
        """
        raise (NotImplementedError)
