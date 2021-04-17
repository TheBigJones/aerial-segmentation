import pytorch_lightning as pl
import torch
import numpy as np

try:
    import wandb
except ModuleNotFoundError:
    pass

from .base import BaseLitModel

class IoU(pl.metrics.classification.IoU):
    """IoU Metric with a hack."""

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Metrics in Pytorch-lightning 1.2+ versions expect preds to be between 0 and 1 else fails with the ValueError:
        "The `preds` should be probabilities, but values were detected outside of [0,1] range."
        This is being tracked as a bug in https://github.com/PyTorchLightning/metrics/issues/60.
        This method just hacks around it by normalizing preds before passing it in.
        Normalized preds are not necessary for accuracy computation as we just care about argmax().
        """
        if preds.min() < 0 or preds.max() > 1:
            preds = torch.nn.functional.softmax(preds, dim=1)
        super().update(preds=preds, target=target)

class UnetLitModel(BaseLitModel):  # pylint: disable=too-many-ancestors
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args=None):
        super().__init__(model, args)
        #TODO don't hardcode num_classes
        self.train_iou = IoU(num_classes=7)
        self.val_iou = IoU(num_classes=7)
        self.test_iou = IoU(num_classes=7)
        self.class_labels = model.class_labels

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.train_iou(logits, y)
        self.log("train_IoU", self.train_iou, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        try:
            original_image = np.moveaxis(x[0].cpu().numpy(),0,-1)
            ground_truth_mask = np.moveaxis(y[0].cpu().numpy(),0,-1)
            #[7,300,300] -> [1,300,300] 1 soll dim argmax
            prediction_mask = torch.argmax(logits[0], dim=0).cpu().numpy()
            self.logger.experiment.log(wandb.Image(original_image, masks={
                "predictions" : {
                    "mask_data" : prediction_mask,
                    "class_labels" : self.class_labels
                },
                "ground_truth" : {
                    "mask_data" : ground_truth_mask,
                    "class_labels": self.class_labels
                }
            }))
        except AttributeError as e:
            print(e)
            pass

        self.log("val_loss", loss, prog_bar=True)
        self.val_iou(logits, y)
        self.log("val_iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        self.test_iou(logits, y)
        self.log("test_iou", self.test_iou, on_step=False, on_epoch=True)
