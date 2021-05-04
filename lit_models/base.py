import argparse
import pytorch_lightning as pl
import torch
import torchmetrics
import numpy as np
import torch.nn.functional as F

try:
    import wandb
except ModuleNotFoundError:
    pass

from pytorch_lightning.metrics.utils import to_onehot

OPTIMIZER = "Adam"
LR = 3e-4
LOSS = "cross_entropy"
ONE_CYCLE_TOTAL_STEPS = 100
MAX_IMGS=15

ALPHA_TVERSKY = 0.7
BETA_TVERSKY = 0.3
GAMMA_TVERSKY = 3./4

class FocalTverskyLoss(pl.LightningModule):
    def __init__(self, num_classes, weight=None, size_average=True, smooth=0.0, alpha=ALPHA_TVERSKY, beta=BETA_TVERSKY, gamma=GAMMA_TVERSKY):
        super().__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        #if preds.min() < 0 or preds.max() > 1:
        preds = torch.nn.functional.softmax(preds, dim=1)

        targets = to_onehot(targets, self.num_classes)

        #True Positives, False Positives & False Negatives
        TP = (preds * targets).sum()
        FP = ((1-targets) * preds).sum()
        FN = (targets * (1-preds)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        FocalTversky = (1 - Tversky)**self.gamma

        return FocalTversky

class F1Loss(FocalTverskyLoss):
    def __init__(self, num_classes, weight=None, size_average=True, smooth=0.0):
        super().__init__(num_classes, weight, size_average, smooth, 0.5, 0.5, 1.0)

class Accuracy(torchmetrics.Accuracy):
    """Accuracy Metric with a hack."""

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Metrics in Pytorch-lightning 1.2+ versions expect preds to be between 0 and 1 else fails with the ValueError:
        "The `preds` should be probabilities, but values were detected outside of [0,1] range."
        This is being tracked as a bug in https://github.com/PyTorchLightning/metrics/issues/60.
        This method just hacks around it by normalizing preds before passing it in.
        Normalized preds are not necessary for accuracy computation as we just care about argmax().
        """
        #if preds.min() < 0 or preds.max() > 1:
        preds = torch.nn.functional.softmax(preds, dim=1)
        super().update(preds=preds, target=target)


class IoU(torchmetrics.IoU):
    """IoU Metric with a hack."""

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Metrics in Pytorch-lightning 1.2+ versions expect preds to be between 0 and 1 else fails with the ValueError:
        "The `preds` should be probabilities, but values were detected outside of [0,1] range."
        This is being tracked as a bug in https://github.com/PyTorchLightning/metrics/issues/60.
        This method just hacks around it by normalizing preds before passing it in.
        Normalized preds are not necessary for accuracy computation as we just care about argmax().
        """
        #if preds.min() < 0 or preds.max() > 1:
        preds = torch.nn.functional.softmax(preds, dim=1)
        super().update(preds=preds, target=target)


class F1(torchmetrics.F1):
    """F1 Metric with a hack."""

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Metrics in Pytorch-lightning 1.2+ versions expect preds to be between 0 and 1 else fails with the ValueError:
        "The `preds` should be probabilities, but values were detected outside of [0,1] range."
        This is being tracked as a bug in https://github.com/PyTorchLightning/metrics/issues/60.
        This method just hacks around it by normalizing preds before passing it in.
        Normalized preds are not necessary for accuracy computation as we just care about argmax().
        """
        #if preds.min() < 0 or preds.max() > 1:
        preds = torch.nn.functional.softmax(preds, dim=1)
        super().update(preds=preds, target=target)


class BaseLitModel(pl.LightningModule):  # pylint: disable=too-many-ancestors
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)

        self.lr = self.args.get("lr", LR)

        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS)
        self.mask_loss = self.args.get("mask_loss", False)
        self.max_imgs = self.args.get("max_imgs", MAX_IMGS)

        self.img_counter = 0

        self.class_labels = model.class_labels
        self.num_classes = len(self.class_labels)

        self.ignore_index = None
        if self.mask_loss:
            self.ignore_index = [k for k in self.class_labels.keys() if self.class_labels[k] == "IGNORE"][0]

        self.loss_weights = torch.ones(self.num_classes)
        if self.mask_loss:
            self.loss_weights[self.ignore_index] = 0.

        loss = self.args.get("loss", LOSS)
        if loss == "tversky":
            self.loss_fn = FocalTverskyLoss(self.num_classes)
        elif loss == "tversky_high_gamma":
            self.loss_fn = FocalTverskyLoss(self.num_classes, gamma=1.5)
        elif loss == "f1":
            self.loss_fn = F1Loss(self.num_classes)
        else:
            self.loss_fn = getattr(torch.nn.functional, loss)
            self.loss_fn.__init__(weight=self.loss_weights)


        self.fgsm_epsilon = self.args.get("fgsm_epsilon", -1.0)

        self.elevation_alpha = self.args.get("elevation_alpha", 0.0)
        self.predict_elevation = (self.elevation_alpha > 0.0)
        if self.predict_elevation:
            self.elevation_loss = torch.nn.MSELoss()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.train_iou = IoU(num_classes=self.num_classes, ignore_index = self.ignore_index)
        self.val_iou = IoU(num_classes=self.num_classes, ignore_index = self.ignore_index)
        self.test_iou = IoU(num_classes=self.num_classes, ignore_index = self.ignore_index)

        self.train_f1 = F1(num_classes=self.num_classes)
        self.val_f1 = F1(num_classes=self.num_classes)
        self.test_f1 = F1(num_classes=self.num_classes)


    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="optimizer class from torch.optim")
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--one_cycle_max_lr", type=float, default=None)
        parser.add_argument("--one_cycle_total_steps", type=int, default=ONE_CYCLE_TOTAL_STEPS)
        parser.add_argument("--loss", type=str, default=LOSS, help="loss function from torch.nn.functional")
        parser.add_argument("--mask_loss", action="store_true", default=False, help="masks ignores from target in lass-calculation")
        parser.add_argument("--max_imgs", type=int, default=MAX_IMGS, help="Maximum number of images to log")
        parser.add_argument("--fgsm_epsilon", type=float, default=-1.0, help="Epsilon for FGSM attack. epsilon <=0.0 means no attack.")
        return parser

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=self.one_cycle_max_lr, total_steps=self.one_cycle_total_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def forward(self, x):
        return self.model(x)

    def calc_loss(self, logits, y):
        loss = self.loss_fn(logits, y)
        return loss

    def fgsm_attack(self, images, labels, elevation_label=None):

        images.requires_grad = True

        logits = self(images)
        if self.predict_elevation:
            elevation = torch.squeeze(logits[:, self.num_classes:, :, :])
            logits = logits[:, :self.num_classes, :, :]

        fgsm_loss = self.calc_loss(logits, labels)
        if self.predict_elevation:
            fgsm_loss += self.elevation_alpha*self.elevation_loss(elevation, elevation_label)

        # Create perturbed images
        sign_data_grad = images.sign()
        perturbed_images = images + self.fgsm_epsilon*sign_data_grad
        perturbed_images = torch.clamp(perturbed_images, 0, 1).detach()

        # Calculate loss on perturbed images
        logits = self(perturbed_images)
        if self.predict_elevation:
            elevation = torch.squeeze(logits[:, self.num_classes:, :, :])
            logits = logits[:, :self.num_classes, :, :]

        fgsm_loss = self.calc_loss(logits, labels)
        if self.predict_elevation:
            fgsm_loss += self.elevation_alpha*self.elevation_loss(elevation, elevation_label)

        return fgsm_loss


    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        if self.predict_elevation:
            x, y, z = batch
        else:
            x, y = batch
        logits = self(x)
        if self.predict_elevation:
            elevation = torch.squeeze(logits[:, self.num_classes:, :, :])
            logits = logits[:, :self.num_classes, :, :]
        loss = self.calc_loss(logits, y)
        if self.predict_elevation:
            loss += self.elevation_alpha*self.elevation_loss(elevation, z)

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.train_iou(logits, y)
        self.log("train_IoU", self.train_iou, on_step=False, on_epoch=True)
        self.train_f1(logits, y)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)
        self.train_acc(logits, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)


        if self.fgsm_epsilon >= 0.0:
            if self.predict_elevation:
                loss += self.fgsm_attack(images=x, labels=y, elevation_label=z)
            else:
                loss += self.fgsm_attack(images=x, labels=y)

        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        if self.predict_elevation:
            x, y, z = batch
        else:
            x, y = batch
        logits = self(x)
        if self.predict_elevation:
            elevation = torch.squeeze(logits[:, self.num_classes:, :, :])
            logits = logits[:, :self.num_classes, :, :]
        loss = self.calc_loss(logits, y)
        if self.predict_elevation:
            loss += self.elevation_alpha*self.elevation_loss(elevation, z)

        self.log("val_loss", loss, prog_bar=True)
        self.val_iou(logits, y)
        self.log("val_iou", self.val_iou, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.val_f1(logits, y)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)
        self.val_acc(logits, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.img_counter += 1
        if self.img_counter <= self.max_imgs:
            return {'x' : x[0].cpu().numpy(), 'y' : y[0].cpu().numpy(), 'logits' : logits[0].cpu()}


    def validation_epoch_end(self, validation_step_outputs):
        wandb_images = []
        self.img_counter = 0
        try:
            for out in validation_step_outputs:
                original_image = np.moveaxis(out['x'], 0, -1)
                ground_truth_mask = out['y']
                # [7,300,300] -> [1,300,300] 1 soll dim argmax
                prediction_mask = torch.argmax(out['logits'], dim=0).numpy()
                wandb_image = wandb.Image(original_image, masks={
                    "predictions": {
                        "mask_data": prediction_mask,
                        "class_labels": self.class_labels
                    },
                    "ground_truth": {
                        "mask_data": ground_truth_mask,
                        "class_labels": self.class_labels
                    }
                })
                wandb_images.append(wandb_image)


            self.logger.experiment.log({"predictions": wandb_images})
        except AttributeError as e:
                print(e)
                pass

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        if self.predict_elevation:
            x, y, z = batch
        else:
            x, y = batch

        logits = self(x)
        if self.predict_elevation:
            elevation = torch.squeeze(logits[:, self.num_classes:, :, :])
            logits = logits[:, :self.num_classes, :, :]
        self.test_iou(logits, y)
        self.log("test_iou", self.test_iou, on_step=False, on_epoch=True)
        self.test_f1(logits, y)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)
        self.test_acc(logits, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
