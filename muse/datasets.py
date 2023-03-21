"""Contains utilities for building, saving, and loading raw piano-roll
datasets."""

import json
import logging
import random
import mido
from pathlib import Path
from progress.bar import Bar


from pianoroll import PianoRoll
from mutopia import parse_rdf_metadata


class Dataset:
    """Container for datasets of PianoRoll objects.

    Args:
        train (list): List of PianoRoll objects for train set.
        test (list): List of PianoRoll objects for test set.
        test (dict): Dataset level metadata.
    """

    def __init__(self, train: list = [], test: list = [], meta_data: dict = {}):
        """Initialises piano-roll dataset."""
        self.train = train
        self.test = test
        self.meta_data = meta_data

    def append(self, piano_roll: PianoRoll, split: str = "train"):
        """Appends PianoRoll to train or test split."""
        if split == "train":
            self.train.append(piano_roll)
        elif split == "test":
            self.test.append(piano_roll)
        else:
            raise (KeyError)

    # TODO: Implement this method.
    def to_json(self, save_path: str = "."):
        """Saves dataset according to specified path.

        Args:
            save_path (str, optional): path to save location. Defaults to '.'.
        """
        raise (NotImplementedError)

    @classmethod
    def build(
        dir: str,
        recur: bool = False,
        div: int = 4,
        meta_tags: list[str] | None = None,
        tt_split: float = 0.9,
    ):
        """Inplace version of build_dataset.

        Args:
            dir (str): directory including .mid or .midi files.
            recur (bool, optional): If True, recursively search directories
                for . Defaults to False.
            div (int): Amount to subdivide each beat in .mid files.
            meta_tags (list[str]): meta_tags to scrape from .rdf files.
            tt_split (float): test-train split ratio.

        Returns:
            Dataset: piano-roll dataset, loaded from dir.
        """
        return build_dataset(dir, recur, meta_tags, tt_split)

    # TODO: Implement this method.
    @classmethod
    def from_json(cls, path: str):
        raise (NotImplementedError)


# TODO:
# - Implement support for recur and meta_tags.
# - Implement a way to filter mid files according to conditions on metadata.
# - Add proper logging instead of print statements.
def build_dataset(
    dir: str,
    recur: bool = True,
    div: int = 4,
    meta_tags: list[str] | None = None,
    tt_split: float = 0.9,
):
    """Builds a piano-roll dataset from a directory containing .mid files.

    Searches through each

    Args:
        dir (str): _description_
        recur (bool, optional): _description_. Defaults to False.
        div (int): Amount to subdivide each beat in .mid files.
        meta_tags (list[str] | None, optional): _description_. Defaults to None.
        tt_split (float, optional): _description_. Defaults to 0.9.

    Returns:
        _type_: _description_
    """

    # Calculate number of .mid files present
    num_mids = 0
    for path in Path(dir).rglob("*.mid"):
        num_mids += 1

    # Generate PianoRoll objects
    p_roll_unsplit = []
    with Bar("Building dataset...", max=num_mids) as bar:
        for path in Path(dir).rglob("*.mid"):
            mid = mido.MidiFile(path)

            try:
                piano_roll = PianoRoll.from_midi(mid, div)
            except Exception as e:
                print(f"\n Error parsing file {path}")
                print(e)
                continue

            piano_roll.add_metadata(parse_rdf_metadata(path.parent))
            piano_roll.add_metadata({"file_name": path.name})
            p_roll_unsplit.append(piano_roll)

            bar.next()

    # For repeatability when building train-test split
    random.seed(42)
    random.shuffle(p_roll_unsplit)
    split_ind = round(tt_split * len(p_roll_unsplit))

    return Dataset(p_roll_unsplit[:split_ind], p_roll_unsplit[split_ind:])


def test():
    dataset = build_dataset("data/raw/mutopia")


if __name__ == "__main__":
    test()
