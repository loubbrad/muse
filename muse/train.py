"""Contains training code for pre-training and fine-tuning using Pytorch
Lightning."""

import re
import argparse
from typing import Optional
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint


from models.model import MuseMaskedLM, ModelConfig
from models.tokenizer import (
    MaskedLMPretrainTokenizer,
    CasualPretrainTokenizer,
    FugueTokenizer,
)
from datasets import TrainDataset


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
            sync_dist=True,  # Not sure what this does
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
            sync_dist=True,  # Not sure what this does
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


def train(
    mode: str,
    checkpoint: Optional[str],
    data: str,
    workers: int,
    gpus: int,
    epochs: int,
):
    lr = 3e-4
    batch_size = 32
    model_config = ModelConfig()

    if mode == "maskedlm-pretrain":
        model_config.use_casual_mask = False
        model_config.drop_p = 0.0
        tokenizer = MaskedLMPretrainTokenizer(model_config)
    elif mode == "casual-pretrain":
        model_config.use_casual_mask = True
        model_config.drop_p = 0.1
        tokenizer = CasualPretrainTokenizer(model_config)
    elif mode == "finetune":
        tokenizer = FugueTokenizer(model_config)
        model_config.drop_p = 0.1
    else:
        raise ValueError

    if isinstance(checkpoint, str) and checkpoint is not None:
        model = MusePretrainLM.load_from_checkpoint(checkpoint)
    elif checkpoint is None:
        model = MusePretrainLM(model_config, lr=lr)

    dataset_train = TrainDataset.from_json(data, tokenizer, key="train")
    dataset_test = TrainDataset.from_json(data, tokenizer, key="test")
    dl_train = DataLoader(
        dataset_train, batch_size=batch_size, num_workers=workers
    )
    dl_test = DataLoader(
        dataset_test, batch_size=batch_size, num_workers=workers
    )

    # See https://shorturl.at/AGHZ3
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{train_loss}-{val_loss}",
        save_top_k=5,
        monitor="val_loss",
        save_weights_only=False,
    )

    a100_re = re.compile(r"[aA]100")
    v100_re = re.compile(r"[vV]100")
    if a100_re.search(torch.cuda.get_device_name(0)):
        prec = "bf16"
    elif v100_re.search(torch.cuda.get_device_name(0)):
        prec = "16-mixed"
    else:
        print("GPU not A100 or V100")
        prec = "16-mixed"

    trainer = pl.Trainer(
        devices=gpus,
        accelerator="gpu",
        precision=prec,
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
    )

    # Train
    trainer.fit(model, dl_train, dl_test)


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
    argp.add_argument(
        "-m",
        "--mode",
        choices=["maskedlm-pretrain", "casual-pretrain", "finetune"],
        required=True,
    )

    argp.add_argument("-c", "--checkpoint")
    argp.add_argument("-d", "--data", type=str)
    argp.add_argument("--workers", type=int, default=1)
    argp.add_argument("--gpus", type=int, default=1)
    argp.add_argument("--epochs", type=int, required=True)
    kwargs = vars(argp.parse_args())

    return kwargs


if __name__ == "__main__":
    kwargs = parse_arguments()
    train(**kwargs)
