import lightning as L
import torchmetrics
from torch import optim, nn, utils, Tensor


class CimatModule(L.LightningModule):
    def __init__(self, model, optimizer, loss_fn):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_accuracy = torchmetrics.Accuracy(task="binary")
        self.valid_accuracy = torchmetrics.Accuracy(task="binary")
        self.train_meaniou = torchmetrics.segmentation.MeanIoU(
            num_classes=2, per_class=True
        )
        self.valid_meaniou = torchmetrics.segmentation.MeanIoU(
            num_classes=2, per_class=True
        )

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self.model(inputs)
        train_loss = self.loss_fn(preds, labels)
        self.train_accuracy(preds, labels)
        self.train_meaniou(preds, labels)
        self.log_dict(
            {
                "train_loss": train_loss,
                "train_acc": self.train_accuracy,
                "train_meaniou": self.train_meaniou,
            },
            prog_bar=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self.model(inputs)
        valid_loss = self.loss_fn(preds, labels)
        self.valid_accuracy(preds, labels)
        self.valid_meaniou(preds, labels)
        self.log_dict(
            {
                "valid_loss": valid_loss,
                "valid_acc": self.valid_accuracy,
                "valid_meaniou": self.valid_meaniou,
            },
            prog_bar=True,
            sync_dist=True,
        )
        return valid_loss

    def configure_optimizers(self):
        return self.optimizer
