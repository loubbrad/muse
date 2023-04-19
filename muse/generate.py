"""Contains function for generating samples (using Gibbs sampling) from fine-
tuned models."""

import math
import json
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
    num_steps = 1000
    neta = 0.35

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
    seq = tokenizer.encode(seq).cuda()
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

        print(n)

    if piano_roll is True:
        return pianoroll.PianoRoll.from_seq(tokenizer.decode(seq))
    else:
        return tokenizer.decode(seq)


def lazy_casual_sample(
    prompt: list,
    num_tokens: int,
    model: MuseMaskedLM,
    tokenizer: Tokenizer,
    piano_roll: bool = True,
):
    idx = len(prompt)
    prompt += [tokenizer.pad_tok] * (tokenizer.max_seq_len - len(prompt))
    seq_enc = tokenizer.encode(prompt).reshape(1, -1).cuda()

    for _ in range(num_tokens):
        logits = model.forward(seq_enc)[0, idx - 1, :]
        probs = torch.nn.functional.softmax(logits, dim=0)
        sample = torch.multinomial(probs, 1)
        seq_enc[0, idx] = sample
        idx += 1

    print(tokenizer.decode(seq_enc[0]))
    if piano_roll is True:
        return pianoroll.PianoRoll.from_seq(tokenizer.decode(seq_enc[0]))
    else:
        return tokenizer.decode(seq_enc[0])


def sample_maskedlm():
    load_path = ""
    model_config = ModelConfig()
    tokenizer = Tokenizer(model_config)
    model = get_torch_module(load_path).cuda()
    model.eval()

    # No prompts
    # num_notes = math.floor((model_config.max_seq_len - 3) / 5)
    # prompt = ["<M>", "<M>", "<M>", "<M>", "<T>"] * num_notes
    # prompt[-1] = "<E>"
    # prompt += ["<P>"] * (model_config.max_seq_len - len(prompt))

    # Load prompts
    with open("data/raw/prompts/fugue_prompts_2048.json") as f:
        prompts = json.load(f)

    for i, prompt in enumerate(prompts):
        gibbs_config = GibbsConfig()
        p_roll = gibbs_unmask(
            prompt,
            model,
            tokenizer,
            gibbs_config,
            piano_roll=True,
        )

        mid = p_roll.to_midi()
        mid.save(f"samples/test{i+1}.mid")


if __name__ == "__main__":
    sample_maskedlm()
