"""Contains utilities for building, saving, and loading raw (and tokenizer)
piano-roll datasets."""

import json
import logging
import random
import mido
import torch.utils
from typing import Callable
from pathlib import Path
from progress.bar import Bar


from pianoroll import PianoRoll, pianoroll_to_midi
from models import ModelConfig, Tokenizer, PretrainTokenizer


class Dataset:
    """Container for datasets of PianoRoll objects.

    Args:
        train (list[PianoRoll]): List of PianoRoll objects for train set.
        test (list[PianoRoll]): List of PianoRoll objects for test set.
        meta_data (dict): Dataset level metadata.
    """

    def __init__(
        self,
        train: list[PianoRoll] = [],
        test: list[PianoRoll] = [],
        meta_data: dict = {},
    ):
        """Initialises piano-roll dataset."""
        self.train = train
        self.test = test
        self.meta_data = meta_data

    def to_train(self, tokenizer: Tokenizer):
        """Returns a TrainDataset object initiated with dataset=self.

        See TrainDataset class for more information.

        Args:
            tokenizer (model.Tokenizer): Tokenizer subclass holding methods for
                PianoRoll to torch.Tensor conversion.
            device (str): Device to send tensors to on __getitem__().
        """
        return TrainDataset(self, tokenizer)

    def to_json(self, save_path: str):
        """Saves dataset according to specified path.

        Args:
            save_path (str): path to save dataset.
        """
        data = {"train": [], "test": [], "meta_data": self.meta_data}

        for entry in self.train:
            data["train"].append(entry.to_dict())
        for entry in self.test:
            data["test"].append(entry.to_dict())

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def from_json(cls, load_path: str):
        """Loads a .json file into a Dataset.

        Args:
            load_path (str): path to load dataset from.

        Returns:
            Dataset: Dataset loaded from .json.
        """

        with open(load_path) as f:
            data = json.load(f)

        meta_data = data["meta_data"]
        train, test = [], []
        for entry in data["train"]:
            train.append(PianoRoll(**entry))
        for entry in data["test"]:
            test.append(PianoRoll(**entry))

        return Dataset(train, test, meta_data)

    @classmethod
    def build(
        cls,
        dir: str,
        recur: bool = False,
        div: int = 4,
        metadata_fn: Callable | None = None,
        filter_fn: Callable | None = None,
        tt_split: float = 0.9,
    ):
        """Builds a piano-roll dataset from a directory containing .mid files.

        Args:
            dir (str): directory including .mid files.
            recur (bool, optional): If True, recursively search directories
                for . Defaults to False.
            div (int): Amount to subdivide each beat in .mid files.
            metadata_fn: (Callable | None, optional): Function to call to scrape
                metadata when a .mid is found. metadata_fn will be given the
                .mid file's path as an argument. It should return a dictionary
                of metadata to add to the PianoRoll. Defaults to None.
            filter_fn: (Callable | None, optional): Function used to filter
                PianoRolls in train and test. It should return True when
                provided a valid PianoRoll, and False otherwise. For an
                example, see filter_instrument in mutopia.py.
            tt_split (float, optional): Ratio for test-train split. Represents
                the proportion to be included in the train split. Defaults to
                0.9.

        Returns:
            Dataset: Dataset of all PianoRoll objects successfully passed from
                .mid in the directory dir.
        """
        return build_dataset(dir, recur, div, metadata_fn, filter_fn, tt_split)


def build_dataset(
    dir: str,
    recur: bool = True,
    div: int = 4,
    metadata_fn: Callable | None = None,
    filter_fn: Callable | None = None,
    tt_split: float = 0.9,
):
    """Builds a piano-roll dataset from a directory containing .mid files.

    Args:
        dir (str): directory including .mid files.
        recur (bool, optional): If True, recursively search directories
            for . Defaults to False.
        div (int): Amount to subdivide each beat in .mid files.
        metadata_fn: (Callable | None, optional): Function to call to scrape
            metadata when a .mid is found. metadata_fn will be given the .mid
            file's path as an argument. It should return a dictionary of
            metadata to add to the PianoRoll. Defaults to None.
        filter_fn: (Callable | None, optional): Function used to filter
            PianoRolls in train and test. It should return True when
            provided a valid PianoRoll, and False otherwise. For an
            example, see filter_instrument in mutopia.py.
        tt_split (float, optional): Ratio for test-train split. Represents the
            proportion to be included in the train split. Defaults to 0.9.

    Returns:
        Dataset: Dataset of all PianoRoll objects successfully passed from
            .mid in the directory dir.
    """

    # Calculate number of .mid files present
    num_mids = 0
    for path in Path(dir).rglob("*.mid"):
        num_mids += 1

    # Generate PianoRoll objects
    p_roll_unsplit = []
    with Bar("Building dataset...", max=num_mids) as bar:
        if recur is True:
            mid_paths = Path(dir).rglob("*.mid")
        else:
            mid_paths = Path(dir).glob("*.mid")

        for path in mid_paths:
            try:
                mid = mido.MidiFile(path)
                piano_roll = PianoRoll.from_midi(mid, div)
            except Exception:
                print("\n")
                logging.error(f"Parsing file at {path} failed.", exc_info=True)
                bar.next()
                continue

            # Add metadata according to metadata_fn
            if metadata_fn is not None:
                piano_roll.add_metadata(metadata_fn(path))
            piano_roll.add_metadata({"file_name": path.name})

            # Filter according to filter_fn
            if filter_fn is None:
                p_roll_unsplit.append(piano_roll)
            elif filter_fn is not None and filter_fn(piano_roll) is True:
                p_roll_unsplit.append(piano_roll)

            bar.next()

    # For repeatability when building train-test split
    random.seed(42)
    random.shuffle(p_roll_unsplit)
    split_ind = round(tt_split * len(p_roll_unsplit))

    return Dataset(p_roll_unsplit[:split_ind], p_roll_unsplit[split_ind:])


class TrainDataset(torch.utils.data.Dataset):
    """PyTorch Dataset class for training data.

    On __init__() uses tokenizer to sequentialise a Dataset (PianoRoll). This
    class is meant as an input for a PyTorch Dataloader object.

    Args:
        dataset (Dataset): Dataset (PianoRoll) for training.
        tokenizer (model.Tokenizer): Tokenizer subclass holding methods for
            PianoRoll to torch.Tensor conversion.
        split (str): Whether to use train or test set.
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: Tokenizer,
        split: str,
    ):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.tok_to_id["<P>"]

        if split == "train":
            self.data = []
            for piano_roll in dataset.train:
                self.data += tokenizer.seq(piano_roll)
        elif split == "test":
            self.data = []
            for piano_roll in dataset.test:
                self.data += tokenizer.seq(piano_roll)
        else:
            raise ValueError("Invalid value for split.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        src, tgt = self.tokenizer.apply(self.data[idx])

        src_enc = self.tokenizer.encode(src)
        tgt_enc = self.tokenizer.encode(tgt)
        mask = torch.where(src_enc == self.pad_id, False, True)

        return src_enc, tgt_enc, mask


def test():
    dataset = Dataset.build("data/raw/miscellaneous", recur=True)
    model_config = ModelConfig()
    tokenizer = PretrainTokenizer(model_config)
    train_dataset = TrainDataset(dataset, tokenizer, split="test")

    print(len(train_dataset))


if __name__ == "__main__":
    test()
