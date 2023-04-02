"""Contains function for generating samples (using Gibbs sampling) from fine-
tuned models."""

import os
import math
import torch
import mido
from dataclasses import dataclass


import pianoroll
from train import get_torch_module
from models.model import ModelConfig, MuseMaskedLM
from models.tokenizer import FinetuneTokenizer


@dataclass
class GibbsConfig:
    alpha_max = 1.0
    alpha_min = 0.05
    num_steps = 250
    neta = 0.75

    temp_max = 1.0
    temp_min = 0.65


# Not tested
def gibbs_unmask(
    seq: list,
    model: MuseMaskedLM,
    tokenizer: FinetuneTokenizer,
    config: GibbsConfig,
    piano_roll: bool = True,
):
    """Un-masks a sequence using gibbs sampling.

    Args:
        seq (list): Sequence to be unmasked.
        model (MuseMaskedLM): Masked token model to sample from.
        tokenizer (FinetuneTokenizer): Tokenizer corresponding to model.
        config (GibbsConfig): Hyperparameters for Gibbs sampling.
        piano_roll (bool): If true gibbs_sample will automatically convert
            the result to a PianoRoll object. Defaults to True.

    Returns:
        Optional[list, PianoRoll]: Un-masked sequence.
    """

    def _gibbs_step(
        src: torch.Tensor,
        model: MuseMaskedLM,
        idx: torch.Tensor,
        temp: float,
    ):
        """Internal function for resampling src using model.

        Args:
            src (torch.Tensor): Tensor to me un-masked.
            model (MuseMaskedLM): Model to resample from.
            idx (torch.Tensor): Indexes to resample.
            temp (float): Temperature to apply to model distribution.

        Returns:
            torch.Tensor: src un-masked.
        """
        logits = (
            model.forward(src.reshape(1, -1)) / temp
        )  # Shape (1, seq_len, vocab_len)
        probs = torch.nn.functional.softmax(
            logits[0, idx, :], dim=1
        )  # Shape (block_size, vocab_len)

        for i in range(idx.shape):
            src[idx[i]] = torch.multinomial(probs[i], 1)

        return src

    def _gibbs_sample(
        seq: list,
        model: MuseMaskedLM,
        tokenizer: FinetuneTokenizer,
        config: GibbsConfig,
    ):
        """Internal function for unmasking a sequence via gibbs sampling.

        Args:
            seq (list): Sequence to be unmasked.
            model (MuseMaskedLM): Masked token model to sample from.
            tokenizer (FinetuneTokenizer): Tokenizer corresponding to
               model.
            config (GibbsConfig): Hyperparameters for Gibbs sampling.

        Returns:
            list: seq un-masked.
        """
        src = torch.tensor(tokenizer.encode(seq))
        mask_id = tokenizer.tok_to_id[tokenizer.mask_tok]
        total_to_mask = torch.sum(src == mask_id).item()

        # Create uniform distribution over initially masked positions.
        dist = torch.where(src == mask_id, 1.0, 0.0)
        dist = dist / torch.linalg.norm(dist)

        # Gibbs sampling
        for n in range(config.num_steps):
            # Calculate temperature and masking probability
            temp = config.temp_max + n * (config.temp_min - config.temp_max) / (
                config.num_steps
            )
            mask_prob = max(
                config.alpha_min,
                config.alpha_max
                - (  # noqa
                    (n * (config.alpha_max - config.alpha_min))
                    / (config.neta * config.num_steps)  # noqa
                ),
            )

            # Randomly mask and resample mask_prob*total_to_mask tokens
            block_size = max(1, math.trunc(mask_prob * total_to_mask))
            idx = torch.multinomial(dist, block_size, replacement=False)
            src[idx] = mask_id

            src = _gibbs_step(src, model, idx, temp)

        return tokenizer.decode(src)

    seq_unmasked = _gibbs_sample(seq, model, tokenizer, config)

    if piano_roll is True:
        return pianoroll.PianoRoll.from_seq(seq_unmasked)
    else:
        return tokenizer.decode(seq_unmasked)


def main():
    load_path = ""

    model_config = ModelConfig()
    tokenizer = FinetuneTokenizer(model_config)
    model = get_torch_module(load_path)

    x = ["<S>"] + ["<M>", "<M>", "<M>", "<M>", "<T>"] * 175
    x.pop()
    x.append("<E>")
    x += ["<P>"] * (model_config.max_seq_len - len(x))

    gibbs_config = GibbsConfig()
    res = gibbs_unmask(x)

    print(res)


if __name__ == "__main__":
    main()
