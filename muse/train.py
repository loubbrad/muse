"""Training code using PyTorch Lightning."""

import torch
from torch import nn as nn
from torch.utils.data import DataLoader
import lightning as pl


from models.model import MuseMaskedLM, ModelConfig
from models.tokenizer import PretrainTokenizer
from datasets import Dataset


class MusePretrainLM(pl.LightningModule):
    """PyTorch Lightning Module for MuseMaskedLM."""

    def __init__(self, model_config: ModelConfig, lr: float):
        super().__init__()

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
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch

        logits = self.model(src).transpose(1, 2)
        loss = self.loss_fn(logits, tgt)

        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


def main():
    model_config = ModelConfig()
    tokenizer = PretrainTokenizer(model_config)

    lr = 3e-4
    batch_size = 16

    dataset = Dataset.from_json("data/processed/chorale_dataset.json")
    dataset_train = dataset.to_train(tokenizer, split="train")
    dataset_val = dataset.to_train(tokenizer, split="test")
    dl_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=4)
    dl_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=4)

    model = MusePretrainLM(model_config, lr=lr)
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        precision="16-mixed",
        enable_checkpointing=False,
        max_epochs=5,
    )

    # Train
    trainer.fit(model, dl_train, dl_val)

    # Predict
    # pred = torch.argmax(logits, dim=2)


if __name__ == "__main__":
    main()
