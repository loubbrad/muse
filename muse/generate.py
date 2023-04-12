"""Contains function for generating samples (using Gibbs sampling) from fine-
tuned models."""

import math
import torch
from dataclasses import dataclass


import pianoroll
from train import get_torch_module
from models.model import ModelConfig, MuseMaskedLM
from models.tokenizer import Tokenizer


@dataclass
class GibbsConfig:
    alpha_max = 1.0
    alpha_min = 0.05
    num_steps = 250
    neta = 0.75

    temp_max = 1.0
    temp_min = 0.65


def gibbs_unmask(
    seq: list,
    model: MuseMaskedLM,
    tokenizer: Tokenizer,
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
    seq = tokenizer.encode(seq).to(device=model.device)
    mask_key = tokenizer.tok_to_id[tokenizer.mask_tok]
    total_to_mask = torch.sum(seq == mask_key).item()
    uniform_dist = torch.where(seq == mask_key, 1.0, 0.0)
    uniform_dist = uniform_dist / torch.linalg.norm(uniform_dist)

    # Hyperparams
    alpha_max = config.alpha_max
    alpha_min = config.alpha_min
    num_steps = config.num_steps
    neta = config.neta
    temp_max = config.temp_max
    temp_min = config.temp_min

    # Gibbs sampling
    for n in range(num_steps):
        # Calc masking rate and temperature
        temp = temp_max + n * ((temp_min - temp_max) / (num_steps))
        mask_prob = max(
            alpha_min,
            alpha_max - ((n * (alpha_max - alpha_min)) / (neta * num_steps)),
        )

        block_size = max(1, math.trunc(mask_prob * total_to_mask))
        idx = torch.multinomial(uniform_dist, block_size, replacement=False)
        seq[idx] = mask_key
        logits = (
            model.forward(seq.reshape(1, -1)) / temp
        )  # Shape (1, seq_len, vocab_len)
        probs = torch.nn.functional.softmax(
            logits[0, idx, :], dim=1
        )  # Shape (block_size, vocab_len)

        for i in range(block_size):
            seq[idx[i]] = torch.multinomial(probs[i], 1)

    if piano_roll is True:
        return pianoroll.PianoRoll.from_seq(tokenizer.decode(seq))
    else:
        return tokenizer.decode(seq)


def main():
    load_path = ""

    model_config = ModelConfig()
    tokenizer = Tokenizer(model_config)
    model = get_torch_module(load_path).cuda()
    model.eval()

    num_notes = math.floor((model_config.max_seq_len - 3) / 5)
    prompt = ["<M>", "<M>", "<M>", "<M>", "<T>"] * num_notes
    prompt[-1] = "<E>"
    prompt += ["<P>"] * (model_config.max_seq_len - len(prompt))

    for i in range(10):
        gibbs_config = GibbsConfig()
        res_dec = gibbs_unmask(
            prompt,
            model,
            tokenizer,
            gibbs_config,
            piano_roll=False,
        )

        print(res_dec)
        print(len(res_dec))

        p_roll = pianoroll.PianoRoll.from_seq(res_dec)
        mid = p_roll.to_midi()
        mid.save(f"samples/muse/test{i}.mid")


if __name__ == "__main__":
    main()
