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


from pianoroll import PianoRoll
from mutopia import parse_rdf_metadata, filter_instrument
from models.model import ModelConfig
from models.tokenizer import (
    Tokenizer,
    MaskedLMPretrainTokenizer,
    FugueTokenizer,
    CasualPretrainTokenizer,
)


class PianoRollDataset:
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

    def to_train(self, tokenizer: Tokenizer, split: str):
        """Returns a TrainDataset object initiated with dataset=self.

        See TrainDataset class for more information.

        Args:
            tokenizer (model.Tokenizer): Tokenizer subclass holding methods for
                PianoRoll to torch.Tensor conversion.
            device (str): Device to send tensors to on __getitem__().
            split (str): Whether to use train or test set.
        """
        return TrainDataset.from_pianoroll_dataset(self, tokenizer, split)

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

        return PianoRollDataset(train, test, meta_data)

    @classmethod
    def build(
        cls,
        dir: str,
        recur: bool = False,
        div: int = 4,
        metadata_fn: Callable | None = None,
        filter_fn: Callable | None = None,
        tt_split: float = 0.95,
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
    filter_num = 0
    parse_err_num = 0
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
                parse_err_num += 1
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
            else:
                filter_num += 1

            bar.next()

    print(
        f"Finished. Failed to parse {parse_err_num} files. Filtered {filter_num} files."
    )

    # For repeatability when building train-test split
    random.seed(42)
    random.shuffle(p_roll_unsplit)
    split_ind = round(tt_split * len(p_roll_unsplit))

    return PianoRollDataset(
        p_roll_unsplit[:split_ind], p_roll_unsplit[split_ind:]
    )


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
        tokenizer: Tokenizer,
        data: list = [],
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        src, tgt = self.tokenizer.apply(self.data[idx])

        src_enc = self.tokenizer.encode(src)
        tgt_enc = self.tokenizer.encode(tgt)

        return src_enc, tgt_enc

    @classmethod
    def from_pianoroll_dataset(
        cls,
        dataset: PianoRollDataset,
        tokenizer: Tokenizer,
        split: str,
    ):
        """Uses tokenizer.seq() to sequentialise dataset."""
        data = []
        if split == "train":
            for piano_roll in dataset.train:
                data += tokenizer.seq(piano_roll)
        elif split == "test":
            for piano_roll in dataset.test:
                data += tokenizer.seq(piano_roll)
        else:
            raise ValueError("Invalid value for split.")

        return TrainDataset(tokenizer, data)

    @classmethod
    def from_json(
        cls,
        load_path: str,
        tokenizer: Tokenizer,
        key: str | None = None,
    ):
        with open(load_path) as f:
            if key is None:
                raw_data = json.load(f)
            else:
                raw_data = json.load(f)[key]

        assert isinstance(raw_data, list), "Loaded data must be a list."
        assert (
            len(raw_data[0]) == tokenizer.max_seq_len
        ), "Sequence len mismatch."

        return TrainDataset(tokenizer, raw_data)


def main():
    name = "fugue"
    # dataset = PianoRollDataset.build(
    #    "data/raw/mutopia",
    #    recur=True,
    #    metadata_fn=parse_rdf_metadata,
    #    filter_fn=filter_instrument,
    # )
    # dataset.to_json("data/processed/mutopia.json")

    dataset = PianoRollDataset.from_json(f"data/raw/counterpoint/{name}.json")
    print("train", len(dataset.train))
    print("test", len(dataset.test))

    model_config = ModelConfig()
    tokenizer = FugueTokenizer(model_config)
    train_dataset = TrainDataset.from_pianoroll_dataset(
        dataset,
        tokenizer,
        split="train",
    )
    print(len(train_dataset))
    test_dataset = TrainDataset.from_pianoroll_dataset(
        dataset,
        tokenizer,
        split="test",
    )
    print(len(test_dataset))

    res = {"train": train_dataset.data, "test": test_dataset.data}
    with open(
        f"data/processed/{name}_{model_config.max_seq_len}_{model_config.stride_len}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(res, f)

    model_config = ModelConfig()
    tokenizer = MaskedLMPretrainTokenizer(model_config)
    train_dataset = TrainDataset.from_json(
        f"{name}_{model_config.max_seq_len}_{model_config.stride_len}.json",
        tokenizer,
        "train",
    )

    print(len(train_dataset))

    while True:
        i = random.randint(1, 200)
        seq_enc = train_dataset[i][1]
        seq_dec = tokenizer.decode(seq_enc)
        p_roll = PianoRoll.from_seq(seq_dec)
        mid = p_roll.to_midi()
        mid.save("test.mid")
        input("...")


if __name__ == "__main__":
    main()
