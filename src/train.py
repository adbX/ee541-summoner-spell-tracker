import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch.cuda
import torch.nn as nn
import torchmetrics
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from torchvision.models.resnet import resnet50
from rich import print as rprint
from rich.table import Table

from model import CustomResNet
from prepare_dataset import LoLImageDataset, get_dataset_stats, print_stats


class SummonerSpellModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet50(weights="ResNet50_Weights.DEFAULT")
        num_filters = self.model.fc.in_features
        layers = list(self.model.children())[:-1]
        self.dropout = nn.Dropout(0.5)
        self.feature_extractor = nn.Sequential(*layers)

        num_target_classes = num_classes
        self.classifier = nn.Linear(num_filters, num_target_classes)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, ignore_index=-1
        )
        self.precision = torchmetrics.Precision(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=-1,
            average="weighted",
        )
        self.recall = torchmetrics.Recall(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=-1,
            average="weighted",
        )
        self.f1_score = torchmetrics.F1Score(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=-1,
            average="weighted",
        )

    def freeze(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        features = self.dropout(features)
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self(x)

        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)
        self.log(
            "train_loss",
            loss,
            batch_size=x.size(0),
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_acc",
            acc,
            batch_size=x.size(0),
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_precision",
            self.precision(y_hat, y),
            batch_size=x.size(0),
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_recall",
            self.recall(y_hat, y),
            batch_size=x.size(0),
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_f1",
            self.f1_score(y_hat, y),
            batch_size=x.size(0),
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self(x)

        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)
        self.log(
            "val_loss",
            acc,
            batch_size=x.size(0),
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_acc",
            acc,
            batch_size=x.size(0),
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_precision",
            self.precision(y_hat, y),
            batch_size=x.size(0),
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_recall",
            self.recall(y_hat, y),
            batch_size=x.size(0),
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_f1",
            self.f1_score(y_hat, y),
            batch_size=x.size(0),
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)
        self.log(
            "test_loss",
            loss,
            batch_size=x.size(0),
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_acc",
            acc,
            batch_size=x.size(0),
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_precision",
            self.precision(y_hat, y),
            batch_size=x.size(0),
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_recall",
            self.recall(y_hat, y),
            batch_size=x.size(0),
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_f1",
            self.f1_score(y_hat, y),
            batch_size=x.size(0),
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=3, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def display_test_metrics(lightning_module):
    table = Table()
    table.add_column("Test metric", justify="center")
    table.add_column("DataLoader 0", justify="center")

    table.add_row("test_acc", f"{lightning_module.accuracy.compute():.4f}")
    table.add_row("test_precision", f"{lightning_module.precision.compute():.4f}")
    table.add_row("test_recall", f"{lightning_module.recall.compute():.4f}")
    table.add_row("test_f1", f"{lightning_module.f1_score.compute():.4f}")

    rprint("\nTest Metrics")
    rprint(table)


def main(hparams):
    seed_everything(42, workers=True)
    transform = transforms.Compose(
        [
            # transforms.Resize((224, 224)),
            # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = LoLImageDataset(root_dir="data/generated", transform=transform)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_stats = get_dataset_stats(dataset, train_set)
    val_stats = get_dataset_stats(dataset, val_set)
    test_stats = get_dataset_stats(dataset, test_set)

    print_stats(train_stats, "Train")
    print_stats(val_stats, "Validation")
    print_stats(test_stats, "Test")

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=8)

    tb_logger = TensorBoardLogger(
        save_dir="tensorboard_logs", name="summoner-spell-tracker"
    )

    num_classes = len(dataset.label_mapping) + 2

    model = SummonerSpellModel(num_classes)
    model.freeze()

    lightning_module = SummonerSpellModel(num_classes)

    print(f"Training on {hparams.devices}...")

    gpu = 1 if torch.cuda.is_available() else None

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, verbose=True, mode="min"
    )

    trainer = pl.Trainer(
        logger=tb_logger,
        max_epochs=hparams.max_epochs,
        accelerator=hparams.accelerator,
        devices=hparams.devices,
        callbacks=[early_stopping],
    )

    trainer.fit(
        lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    print("Testing...")

    trainer.test(dataloaders=test_loader, verbose=False, ckpt_path="best", model=model)

    display_test_metrics(lightning_module)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    main(args)
