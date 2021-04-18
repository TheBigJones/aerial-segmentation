import pytorch_lightning as pl
import torch
import numpy as np

try:
    import wandb
except ModuleNotFoundError:
    pass

from .base import BaseLitModel, IoU


class UnetLitModel(BaseLitModel):  # pylint: disable=too-many-ancestors
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args=None):
        super().__init__(model, args)
        ignore_index = None
        if self.mask_loss:
            ignore_index = [k for k in self.class_labels.keys() if self.class_labels[k] == "IGNORE"][0]
        self.train_iou = IoU(num_classes=self.num_classes, ignore_index = ignore_index)
        self.val_iou = IoU(num_classes=self.num_classes, ignore_index = ignore_index)
        self.test_iou = IoU(num_classes=self.num_classes, ignore_index = ignore_index)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        loss = self.calc_loss(logits, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.train_iou(logits, y)
        self.log("train_IoU", self.train_iou, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        loss = self.calc_loss(logits, y)

        self.log("val_loss", loss, prog_bar=True)
        self.val_iou(logits, y)
        self.log("val_iou", self.val_iou, on_step=False,
                 on_epoch=True, prog_bar=True)
        return {'x' : x[0], 'y' : y[0], 'logits' : logits[0]}

    def validation_epoch_end(self, validation_step_outputs):
        wandb_images = []
        for out in validation_step_outputs:
            try:
                original_image = np.moveaxis(out['x'].cpu().numpy(), 0, -1)
                ground_truth_mask = out['y'].cpu().numpy()
                # [7,300,300] -> [1,300,300] 1 soll dim argmax
                prediction_mask = torch.argmax(out['logits'], dim=0).cpu().numpy()
                wandb_images.append(wandb.Image(original_image, masks={
                    "predictions": {
                        "mask_data": prediction_mask,
                        "class_labels": self.class_labels
                    },
                    "ground_truth": {
                        "mask_data": ground_truth_mask,
                        "class_labels": self.class_labels
                    }
                }))

            except AttributeError as e:
                print(e)
                pass
        self.logger.experiment.log({"predictions": wandb_images})

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        self.test_iou(logits, y)
        self.log("test_iou", self.test_iou, on_step=False, on_epoch=True)
