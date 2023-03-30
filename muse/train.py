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


def main():
    lr = 3e-3
    batch_size = 16

    model_config = ModelConfig()
    tokenizer = PretrainTokenizer(model_config)
    model = MusePretrainLM(model_config, lr=lr)

    dataset = Dataset.from_json("data/processed/chorale_dataset.json")
    dataset_train = dataset.to_train(tokenizer, split="train")
    dataset_val = dataset.to_train(tokenizer, split="test")
    dl_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=4)
    dl_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=4)

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        precision="16-mixed",
        enable_checkpointing=False,
        max_epochs=100,
    )

    # Train
    trainer.fit(model, dl_train, dl_val)


def run_overfit_batch():
    """Over-fits a single batch. Needed as data augmentation is on by default.
    This function is messy and probably won't work. Only used for development.
    """

    class NoAugDataset(torch.utils.data.Dataset):
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    lr = 3e-3
    model_config = ModelConfig()
    tokenizer = PretrainTokenizer(model_config)
    model = MusePretrainLM(model_config, lr=lr)

    dataset = Dataset.from_json("data/processed/chorale_dataset.json")
    dataset = dataset.to_train(tokenizer, split="train")

    x_tmp, y_tmp = dataset[0]
    x, y = [x_tmp], [y_tmp]
    for i in range(1, 16):
        x_tmp, y_tmp = dataset[i]
        x.append(x_tmp)
        y.append(y_tmp)

    x = torch.stack(x, dim=0)
    y = torch.stack(y, dim=0)

    of_dataset_train = NoAugDataset(x[:8], y[:8])
    of_dataloader_train = DataLoader(of_dataset_train, batch_size=8)
    of_dataset_val = NoAugDataset(x[8:], y[8:])
    of_dataloader_val = DataLoader(of_dataset_val, batch_size=8)

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        precision="16-mixed",
        enable_checkpointing=False,
        max_epochs=1000,
        overfit_batches=True,
    )

    # Train
    trainer.fit(model, of_dataloader_train, of_dataloader_val)


if __name__ == "__main__":
    main()
