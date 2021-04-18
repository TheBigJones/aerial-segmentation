import pytorch_lightning as pl
import torch
import numpy as np

try:
    import wandb
except ModuleNotFoundError:
    pass

from .base import BaseLitModel


class UnetLitModel(BaseLitModel):  # pylint: disable=too-many-ancestors
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args=None):
        super().__init__(model, args)
        self.train_iou = IoU(num_classes=self.num_classes)
        self.val_iou = IoU(num_classes=self.num_classes)
        self.test_iou = IoU(num_classes=self.num_classes)


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
        try:
            original_image = np.moveaxis(x[0].cpu().numpy(),0,-1)
            ground_truth_mask = np.moveaxis(y[0].cpu().numpy(),0,-1)
            #[7,300,300] -> [1,300,300] 1 soll dim argmax
            prediction_mask = torch.argmax(logits[0], dim=0).cpu().numpy()
            wandb_image = wandb.Image(original_image, masks={
                "predictions" : {
                    "mask_data" : prediction_mask,
                    "class_labels" : self.class_labels
                },
                "ground_truth" : {
                    "mask_data" : ground_truth_mask,
                    "class_labels": self.class_labels
                }
            })
        except AttributeError as e:
            print(e)
            pass
        self.log("val_loss", loss, prog_bar=True)
        self.val_iou(logits, y)
        self.log("val_iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True)
        return {'wandb_image' : wandb_image}

    def validation_epoch_end(self, validation_step_outputs):
        wandb_images = []
        for out in validation_step_outputs:
            wandb_images.append(out["wandb_image"])
        
        self.logger.experiment.log({"predictions": wandb_images})
            

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        self.test_iou(logits, y)
        self.log("test_iou", self.test_iou, on_step=False, on_epoch=True)
