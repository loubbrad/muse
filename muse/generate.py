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
    alpha_max = 0.15
    alpha_min = 0.05
    num_steps = 1000
    neta = 0.3

    temp_max = 1.0
    temp_min = 0.6


def gibbs_unmask(
    seq: list,
    mask_ids: list,
    model: MuseMaskedLM,
    tokenizer: Tokenizer,
    config: GibbsConfig,
    piano_roll: bool = True,
):
    """Un-masks a sequence using gibbs sampling.

    Args:
        seq (list): Sequence to be unmasked.
        mask_ids: Not fin. ***
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
    total_to_mask = sum(mask_ids)
    dist = torch.tensor(mask_ids, dtype=torch.float)
    dist = dist / total_to_mask

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
        idx = torch.multinomial(dist, block_size, replacement=False)
        seq[idx] = mask_key

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model.forward(seq.reshape(1, -1)) / temp
            probs = torch.nn.functional.softmax(logits[0, idx, :], dim=1)

        for i in range(block_size):
            seq[idx[i]] = torch.multinomial(probs[i], 1)

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


def sample_fugue():
    load_path = "models/params.ckpt"
    model_config = ModelConfig()
    gibbs_config = GibbsConfig()
    tokenizer = Tokenizer(model_config)
    model = get_torch_module(load_path).cuda()
    model.eval()

    # Load prompts
    with open("data/prompts.json") as f:
        prompts = json.load(f)

    num_bars = 4
    for i, prompt in enumerate(prompts):
        assert tokenizer.unk_tok not in tokenizer.decode(
            tokenizer.encode(prompt)
        ), "unk_tok present in prompt"

        prompt_len = 5 * 4 * 4 * num_bars

        def _mask_id_fn(pos: int, tok: str | int) -> int:
            if pos > prompt_len and (isinstance(tok, int) or tok == "<O>"):
                return 1
            else:
                return 0

        mask_ids = [_mask_id_fn(i, tok) for i, tok in enumerate(prompt)]
        res = gibbs_unmask(
            prompt,
            mask_ids,
            model,
            tokenizer,
            gibbs_config,
            piano_roll=False,
        )
        
        print(f'Done {i+1}/{len(prompts)}')
        res_p_roll = pianoroll.PianoRoll.from_seq(res)
        prompt_p_roll = pianoroll.PianoRoll.from_seq(prompt)
        res_mid = res_p_roll.to_midi()
        prompt_mid = prompt_p_roll.to_midi()
        res_mid.save(f"samples/res{i+1}.mid")
        prompt_mid.save(f"samples/prompt{i+1}.mid")


if __name__ == "__main__":
    sample_fugue()
