import lightning as L
from torch import optim, nn, utils, Tensor


class CimatModule(L.LightningModule):
    def __init__(self, model, optimizer, loss_fn):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def training_step(self, batch, batch_idx):
        x, _ = batch
        z = self.model(x)
        loss = loss_fn(z, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer
