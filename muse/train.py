"""Contains training code for pre-training and fine-tuning using Pytorch
Lightning."""

import argparse
from typing import Optional
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint


from models.model import MuseMaskedLM, ModelConfig
from models.tokenizer import PretrainTokenizer, FinetuneTokenizer
from datasets import Dataset


class MusePretrainLM(pl.LightningModule):
    """PyTorch Lightning Module for MuseMaskedLM."""

    def __init__(self, model_config: ModelConfig, lr: float):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.model = MuseMaskedLM(model_config)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=model_config.pad_id)

    def forward(self, src: torch.Tensor):
        return self.model(src)

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        logits = self.model(src).transpose(1, 2)
        loss = self.loss_fn(logits, tgt)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        logits = self.model(src).transpose(1, 2)
        loss = self.loss_fn(logits, tgt).item()

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


def train(mode: str, checkpoint: Optional[str], epochs: int):
    lr = 1e-4
    batch_size = 64
    model_config = ModelConfig()

    if mode == "pt":
        tokenizer = PretrainTokenizer(model_config)
    elif mode == "ft":
        tokenizer = FinetuneTokenizer(model_config)
    else:
        raise ValueError

    if isinstance(checkpoint, str) and checkpoint is not None:
        model = MusePretrainLM.load_from_checkpoint(checkpoint)
    elif checkpoint is None:
        model = MusePretrainLM(model_config, lr=lr)

    dataset = Dataset.from_json("data/processed/cpoint_chorales.json")
    dataset_train = dataset.to_train(tokenizer, split="train")
    dataset_val = dataset.to_train(tokenizer, split="test")
    dl_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=4)
    dl_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=4)

    # DEBUG
    # import pianoroll

    # x = tokenizer.decode(dataset_train[60][1])
    # print(x)
    # roll = pianoroll.PianoRoll.from_seq(x)
    # print(roll.roll)
    # midi = roll.to_midi()
    # midi.save("test.mid")
    # raise Exception

    # See https://shorturl.at/AGHZ3
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{train_loss}-{val_loss}",
        save_top_k=5,
        monitor="train_loss",
        save_weights_only=True,
    )

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        precision="16-mixed",
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
    )

    # Train
    trainer.fit(model, dl_train, dl_val)


def get_torch_module(load_path: str):
    """Extracts the PyTorch module from a checkpointed Lightning module.

    Args:
        load_path (str): Load path for checkpointed Lightning module.

    Returns:
        nn.Module: Module extracted from self.module
    """
    lightning_module = MusePretrainLM.load_from_checkpoint(load_path)

    return lightning_module.model


def parse_arguments():
    argp = argparse.ArgumentParser()
    argp.add_argument("-m", "--mode", choices=["pt", "ft"])
    argp.add_argument("-c", "--checkpoint")
    argp.add_argument("--epochs", type=int)
    kwargs = vars(argp.parse_args())

    return kwargs


if __name__ == "__main__":
    kwargs = parse_arguments()
    train(**kwargs)
