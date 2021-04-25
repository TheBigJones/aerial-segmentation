import pytorch_lightning as pl
import torch
import numpy as np


from .base import BaseLitModel


class UnetLitModel(BaseLitModel):  # pylint: disable=too-many-ancestors
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args=None):
        super().__init__(model, args)
