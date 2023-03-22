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

        self.eos_tok = "<S>"
        self.bos_tok = "<E>"
        self.mask_tok = "<M>"
        self.pad_tok = "<P>"
        self.unk_tok = "<U>"
        self.off_tok = "<O>"

        self.time_tok = "<T>"
        self.cls_tok = "<CLS>"

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
        model_config.mask_id = self.tok_to_id[self.mask_tok]
        model_config.pad_id = self.tok_to_id[self.pad_tok]

    def seq(self):
        pass

    def encode(self):
        pass

    def decode(self):
        pass

    def apply(self):
        pass


# TODO: Test this.
class PretrainTokenizer(Tokenizer):
    """Tokenizes and sequentialises PianoRoll objects for pre-training.

    Args:
        model_config (model.ModelConfig): Config for model.
        device (str): Device to sent PyTorch tensors to send torch.tensors to.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        return_tensors: bool = True,
        note_off_rate: float = 0.05,
    ):
        super.__init__(model_config, return_tensors)

        self.note_off_rate = note_off_rate

    def seq(self, piano_roll: PianoRoll):
        """Sequentialise a PianoRoll object into a list (for pre-training).

        Args:
            piano_roll (PianoRoll): piano-roll to be made sequential.

        Returns:
            list(tuple): A collection of sequences, of the form (seq, pos_ids)
            where seq is a sequentialised piano-roll and pos_ids is the
            corresponding position ids for position encoding.
        """

        def _get_seq(roll: list, beginning: bool = False):
            """Sequentialises roll backwards, modifying roll inplace."""
            # Add bos token if beginning is True
            if beginning is True:
                roll.insert(0, [self.bos_tok])

            # Sequentialise up to self.max_seq_len
            pos = 0
            seq = []
            pos_ids = []
            # -2 accounts for a possible off_tok and eos_tok
            while len(seq) + len(roll[0]) <= self.max_seq_len - 2:
                for note in roll.pop(0):  # Modifies roll inplace
                    seq.append(note)
                    pos_ids.append(pos)

                # Randomly add note-offs
                if random.uniform(0, 1) < self.note_off_rate:
                    seq.append(self.off_tok)
                    pos_ids.append(pos)

                pos += 1

            # If end of piano-roll, append end of sequence
            if not roll:
                seq.append(self.eos_tok)
                pos_ids.append(pos)

            # Pad to self.max_seq_len
            seq = seq + [self.pad_tok] * (self.max_seq_len - len(seq))

            return seq, pos

        roll = copy.deepcopy(piano_roll.roll)
        sequences = [_get_seq(roll, beginning=True)]

        while roll:
            sequences.append(_get_seq(roll, beginning=False))

        return sequences

    def encode(self, seq: list, pos_ids: list):
        """Encodes a sequentialised piano-roll.

        Args:
            seq (list[int|str]): Piano-roll sequence to be encoded.
            pos_ids (list): Position-ids.

        Returns:
            torch.Tensor/list: Encoded sequence. Return type dependent on
            self.return_tensors.
        """

        def enc_fn(tok):
            return self.tok_to_id.get(tok, self.tok_to_id[self.unk_tok])

        if self.return_tensors is True:
            seq_enc = torch.tensor(list(map(enc_fn, seq)))
            pos_ids_enc = torch.tensor(pos_ids, dtype=torch.long)
        else:
            seq_enc = list(map(enc_fn, seq))
            pos_ids_enc = copy.copy(pos_ids)

        return seq_enc, pos_ids_enc

    def decode(
        self,
        seq: Union[torch.Tensor, list],
        pos_ids: Union[torch.Tensor, list, None] = None,
    ):
        """Decodes a torch.Tensor/list.

        Args:
            seq (Union[torch.Tensor, list]): Sequences to be decoded.
            pos_ids (Union[torch.Tensor, list, None], optional): Position ids,
                possibly to be returned from torch.Tensor to list.

        Returns:
            list: Decoded sequence as a list.
            Optional[list, None]: Position ids.
        """

        def dec_fn(id):
            return self.id_to_tok.get(id, self.unk_tok)

        if isinstance(seq, torch.Tensor):
            seq_dec = map(dec_fn, seq.tolist())
        else:
            seq_dec = map(dec_fn, seq)

        if isinstance(pos_ids, torch.Tensor):
            pos_ids_dec = pos_ids.tolist()
            return seq_dec, pos_ids_dec
        else:
            return seq_dec, pos_ids

    def apply(self, seq: list, mask_p: float, mask_special: bool = False):
        """Performs random masking (in place) on piano-roll sequence.

        Args:
            seq (list): Sequences to be randomly masked.
            mask_p (float): Probability that a token should be masked.
            mask_special (bool): Whether to mask special tokens.

        Returns:
            list: Sequences after appropriate masking.
        """

        if mask_special is True:
            exclude_toks = []
        else:
            exclude_toks = self.vocab_special

        for idx, tok in enumerate(seq):
            if tok not in exclude_toks and random.uniform(0, 1) < mask_p:
                seq[idx] = self.mask_tok

        return seq


class FinetuneTokenizer(Tokenizer):
    """Tokenizes and sequentialises PianoRoll objects for pre-training.

    Args:
        model_config (model.ModelConfig): Config for model.
        device (str): Device to sent PyTorch tensors to send torch.tensors to.
    """

    def __init__(self, model_config: ModelConfig, device: str):
        super().__init__()  # Needed?
        raise NotImplementedError

    def seq(self):
        raise NotImplementedError

    def encode(self):
        raise NotImplementedError

    def decode(self):
        raise NotImplementedError

    def apply(self):
        raise NotImplementedError
