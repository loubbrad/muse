"""Contains utilities for building, saving, and loading raw piano-roll
datasets."""

import json
import glob


from .pianoroll import PianoRoll


# TODO: Implement class properly
class Dataset:
    """Container for datasets of PianoRoll objects.

    Args:
        path (str, optional): Path to .json for dataset.
        name (str, optional): Name of dataset corresponding to pre-set
            save location.
    """

    def __init__(self, path: str | None, name: str | None):
        """Initialises PianoRoll with data and metadata."""
        raise (NotImplementedError)

    def save(self, save_path: str = "."):
        """Saves dataset according to specified path.

        Args:
            save_path (str, optional): path to save location. Defaults to '.'.
        """
        raise (NotImplementedError)

    # Needed?
    @classmethod
    def build(dir: str, recursive: bool = False, tt_split: float = 0.9):
        """Inplace version of build_dataset.

        Args:
            dir (str): directory including .mid or .midi files.
            recursive (bool, optional): If True, recursively search directories
                for . Defaults to False.
            tt_split (float): test-train split ratio ()

        Returns:
            Dataset: piano-roll dataset, loaded from dir.
        """
        return build_dataset(dir, recursive)


def build_dataset(dir: str, recursive: bool = False):
    """_summary_

    Args:
        dir (str): _description_
        recursive (bool, optional): _description_. Defaults to False.
    """
    pass
