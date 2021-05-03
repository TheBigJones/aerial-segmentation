from models.unet import Unet
from lit_models import BaseLitModel
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

class SegModel:
    """
    Class to run inference at the end of testing
    """

    def __init__(self, checkpoint_path, model, args):
        self.lit_model = BaseLitModel.load_from_checkpoint(
                checkpoint_path=checkpoint_path, args=args, model=model
            )
        self.lit_model.eval()
        #self.scripted_model = self.lit_model.to_torchscript(method="script", file_path=None)
        self.model = model

    @torch.no_grad()
    def predict(self, images):
        """
        Prediction function for benchmark prediciton at the end of the training process
        """
        #logits = self.scripted_model(images)
        logits = self.lit_model(images)
        if self.model.predict_elevation:
            logits, elevation = torch.split(logits, self.model.num_classes-1, dim = -3)
        preds = F.softmax(logits, dim=-3)
        return preds
