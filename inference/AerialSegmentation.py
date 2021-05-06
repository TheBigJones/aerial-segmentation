import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import math
from pathlib import Path
import json
import argparse

from models.unet import Unet
from lit_models import BaseLitModel
from data import AerialData

CONFIG_AND_WEIGHTS_DIRNAME = Path(__file__).resolve().parents[1] / "artifacts"

class SegModel:
    """
    Class to run inference at the end of testing
    """

    def __init__(self, checkpoint_path=None, model=None, args=None, run_id=None):
        if model is None:
          if run_id is None:
            raise Exception("run_id is required")
          data = AerialData()
          with open(CONFIG_AND_WEIGHTS_DIRNAME / f"aerialdata_{run_id}" / "config.json", "r") as file:
            config = json.load(file)
          args = argparse.Namespace(**config)
          model = Unet(data_config=data.config(), args=args)
          checkpoint_path = CONFIG_AND_WEIGHTS_DIRNAME / f"aerialdata_{run_id}" / "model.pt"
    
        self.lit_model = BaseLitModel.load_from_checkpoint(
                  checkpoint_path=checkpoint_path, args=args, model=model
              )
        self.model = model
        #self.scripted_model = self.lit_model.to_torchscript(method="script", file_path=None)
        self.lit_model.eval()

    def set_image_size(self, image_size):
        encoder_depth = self.model.args["encoder_depth"]
        self.model.image_size = image_size
        self.model.padding = (math.ceil(self.model.image_size / 2**(encoder_depth))*2**(encoder_depth)-self.model.image_size)//2

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
