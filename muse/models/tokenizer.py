"""Includes tokenizer for converting piano-roll to PyTorch tensors for
pre-training, fine-tuning, and inference."""

import random
import copy
import torch
from typing import Union


from models.model import ModelConfig
from pianoroll import PianoRoll


# TODO: Find out how to properly implement this abstract class.
class Tokenizer:
    """Abstract class for PianoRoll Tokenizers."""

    def __init__(self, model_config: ModelConfig, return_tensors: bool = True):
        self.max_seq_len = model_config.max_seq_len
        self.return_tensors = return_tensors

        self.eos_tok = "<E>"
        self.bos_tok = "<S>"
        self.mask_tok = "<M>"
        self.pad_tok = "<P>"
        self.unk_tok = "<U>"
        self.off_tok = "<O>"

        self.time_tok = "<T>"
        self.cls_tok = "<SEP>"

        self.vocab_special = [
            self.eos_tok,
            self.bos_tok,
            self.mask_tok,
            self.pad_tok,
            self.unk_tok,
            self.off_tok,
            self.time_tok,
            self.cls_tok,
        ]

        self.vocab = [i for i in range(0, 128)] + self.vocab_special

        self.tok_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_tok = {i: tok for i, tok in enumerate(self.vocab)}

        model_config.vocab_size = len(self.tok_to_id)
        model_config.pad_id = self.tok_to_id[self.pad_tok]
        model_config.mask_id = self.tok_to_id[self.mask_tok]

    def seq(self):
        pass

    def encode(self):
        pass

    def decode(self):
        pass

    def apply(self):
        pass


class PretrainTokenizer(Tokenizer):
    """Tokenizes and sequentialises PianoRoll objects for pre-training.

    Args:
        model_config (model.ModelConfig): Config for model.
        device (str): Device to sent PyTorch tensors to send torch.tensors to.
        note_off_rate (float): Rate to randomly add masked off-notes.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        return_tensors: bool = True,
        note_off_rate: float = 0.05,
    ):
        super().__init__(model_config, return_tensors)

        self.note_off_rate = note_off_rate

    # TODO: Remove starting silence.
    def seq(self, piano_roll: PianoRoll):
        """Sequentialise a PianoRoll object into a list (for pre-training).

        Args:
            piano_roll (PianoRoll): piano-roll to be made sequential.

        Returns:
            list[list]: A collection of lists of sequentialised piano-roll.
        """

        def _get_seq(roll: list, beginning: bool = False):
            """Sequentialises roll backwards, modifying roll inplace."""
            # Add bos token if beginning is True
            if beginning is True:
                roll.insert(0, [self.bos_tok])

            # Sequentialise up to self.max_seq_len
            seq = []
            # -3 counts for a possible off_tok, time_tok, and eos_tok
            while roll and (len(seq) + len(roll[0]) <= self.max_seq_len - 3):
                for note in roll.pop(0):  # Modifies roll inplace
                    seq.append(note)

                # Randomly add note-offs
                if random.uniform(0, 1) < self.note_off_rate:
                    seq.append(self.off_tok)

                seq.append(self.time_tok)

            # If end of piano-roll, append end of sequence
            if not roll:
                seq.pop()  # Remove last time_tok
                seq.append(self.eos_tok)

            # Pad to self.max_seq_len
            seq = seq + [self.pad_tok] * (self.max_seq_len - len(seq))

            return seq

        roll = copy.deepcopy(piano_roll.roll)
        sequences = [_get_seq(roll, beginning=True)]

        while roll:
            sequences.append(_get_seq(roll, beginning=False))

        return sequences

    def encode(self, seq: list):
        """Encodes a sequentialised piano-roll.

        Args:
            seq (list[int|str]): Piano-roll sequence to be encoded.

        Returns:
            torch.Tensor/list: Encoded sequence. Return type dependent on
            self.return_tensors.
        """

        def enc_fn(tok):
            return self.tok_to_id.get(tok, self.tok_to_id[self.unk_tok])

        if self.return_tensors is True:
            seq_enc = torch.tensor([enc_fn(tok) for tok in seq])
        else:
            seq_enc = [enc_fn(tok) for tok in seq]

        return seq_enc

    def decode(self, seq: Union[torch.Tensor, list]):
        """Decodes a torch.Tensor/list.

        Args:
            seq (Union[torch.Tensor, list]): Sequences to be decoded.

        Returns:
            list: Decoded sequence as a list.
        """

        def dec_fn(id):
            return self.id_to_tok.get(id, self.unk_tok)

        if isinstance(seq, torch.Tensor):
            seq_dec = [dec_fn(idx) for idx in seq.tolist()]
        else:
            seq_dec = [dec_fn(idx) for idx in seq]

        return seq_dec

    def apply(
        self,
        seq: list,
        mask_p: float = 0.15,
        pitch_aug_range: int = 6,
    ):
        """Applies random masking (in place) on piano-roll sequence.

        Args:
            seq (list): Sequences to be randomly masked.
            mask_p (float): Probability that a token should be masked.
            pitch_aug_range (bool): Range to randomly augment all notes by.

        Returns:
            list: Sequences after appropriate masking.
        """

        def _mask_aug_chord(chord: list, src: list, tgt: list):
            """Appends chord to src and tgt."""
            for tok in chord:
                if isinstance(tok, int):
                    if random.uniform(0, 1) < mask_p:
                        src.append(self.mask_tok)
                        tgt.append(tok + pitch_aug)
                    else:
                        src.append(tok + pitch_aug)
                        tgt.append(tok + pitch_aug)
                elif tok == self.off_tok:
                    src.append(self.mask_tok)
                    tgt.append(self.off_tok)

        src, tgt = [], []
        pitch_aug = random.randint(-pitch_aug_range, pitch_aug_range)

        idx = 0
        while idx < self.max_seq_len:
            # Load current chord into buffer
            buffer = []
            while (
                seq[idx] != self.time_tok
                and seq[idx] != self.eos_tok
                and seq[idx] != self.bos_tok
                and seq[idx] != self.pad_tok
            ):
                buffer.append(seq[idx])
                idx += 1

            # Perform shuffling, pitch augmenting, and masking
            if buffer:
                random.shuffle(buffer)  # Randomly shuffle chord
                _mask_aug_chord(buffer, src, tgt)

            # Append time_tok, bos_tok, eos_tok, or pad_tok
            src.append(seq[idx])
            tgt.append(seq[idx])
            idx += 1

        return src, tgt


class FinetuneTokenizer(Tokenizer):
    """Tokenizes and sequentialises PianoRoll objects for pre-training.

    Args:
        model_config (model.ModelConfig): Config for model.
        device (str): Device to sent PyTorch tensors to send torch.tensors to.
    """

    def __init__(self, model_config: ModelConfig, device: str):
        super().__init__()
        raise NotImplementedError

    def seq(self):
        raise NotImplementedError

    def encode(self):
        raise NotImplementedError

    def decode(self):
        raise NotImplementedError

    def apply(self):
        raise NotImplementedError
