import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy
from torchvision import models
from transformers import ViTForImageClassification


class KenyanFood13Classifier(L.LightningModule):
    """
    KenyanFood13Classifier is the model class for the Kenyan Food 13 Classifier.
    """

    def __init__(
        self,
        num_classes: int = 13,
        weights: str = "DEFAULT",
        unfreeze_layers: int = 4,
        learning_rate: float = 0.001,
    ) -> None:
        """
        Initializes the KenyanFood13Classifier.

        Args:
            num_classes (int, optional): number of classes. Defaults to 13.
            weights (str, optional): weights. Defaults to "DEFAULT".
            learning_rate (float, optional): learning rate. Defaults to 0.001.

        Returns:
            None
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        for param in self.model.parameters():
            param.requires_grad = False

        for i in range(1, self.hparams.unfreeze_layers):
            for param in self.model.vit.encoder.layer[-i].parameters():
                param.requires_grad = True

        self.mean_train_accuracy = MulticlassAccuracy(
            num_classes=self.hparams.num_classes,
            average="micro",
        )
        self.mean_val_accuracy = MulticlassAccuracy(
            num_classes=self.hparams.num_classes,
            average="micro",
        )

        self.mean_train_loss = MeanMetric()
        self.mean_val_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        output = self.model(x)
        return output.logits

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Training step of the model.

        Args:
            batch (torch.Tensor): input tensor.
            batch_idx (int): batch index.

        Returns:
            torch.Tensor: loss tensor.
        """
        data, target = batch
        output = self(data)
        loss = F.cross_entropy(output, target)
        pred_batch = output.detach().argmax(dim=1)

        # Update metrics
        size_of_batch = data.size(0)
        self.mean_train_loss(loss, weight=size_of_batch)
        self.mean_train_accuracy(pred_batch, target)

        self.log(
            "train/batch_loss",
            self.mean_train_loss,
            logger=True,
        )
        self.log(
            "train/batch_acc",
            self.mean_train_accuracy,
            logger=True,
        )

        return loss

    def validation_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        """
        Validation step of the model.

        Args:
            batch (torch.Tensor): input tensor.
            batch_idx (int): batch index.

        Returns:
            None
        """
        data, target = batch
        output = self(data)
        loss = F.cross_entropy(output, target)
        pred_batch = output.detach().argmax(dim=1)

        # Update metrics
        size_of_batch = data.size(0)
        self.mean_val_loss(loss, weight=size_of_batch)
        self.mean_val_accuracy(pred_batch, target)

    def on_train_epoch_end(self) -> None:
        """
        Called at the end of each training epoch.

        Returns:
            None
        """
        self.log(
            "train/loss",
            self.mean_train_loss,
            logger=True,
        )
        self.log(
            "train/acc",
            self.mean_train_accuracy,
            logger=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of each validation epoch.

        Returns:
            None
        """
        self.log(
            "val/loss",
            self.mean_val_loss,
            logger=True,
        )
        self.log(
            "val/acc",
            self.mean_val_accuracy,
            logger=True,
            prog_bar=True,
        )

        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            logger=True,
            prog_bar=True,
        )

    def configure_optimizers(self) -> optim.Optimizer:
        """
        Configures the optimizer.

        Returns:
            optim.Optimizer: optimizer
        """
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=3,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/acc",
        }
