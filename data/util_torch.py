"""Base Dataset class."""
from typing import Any, Callable, Dict, Sequence, Tuple, Union
import torch
from PIL import Image
import numpy as np
import random

SequenceOrTensor = Union[Sequence, torch.Tensor]


def load_img(fname, shape):
    image = Image.open(fname)
    image = image.resize(shape)
    image = np.transpose(np.array(image), (1,0,2))
    return image

def load_mask(fname, shape):
    mask = Image.open(fname)
    mask = mask.resize(shape, Image.NEAREST)
    mask = np.transpose(np.array(mask), (1,0,2))
    return mask

def load_elevation(fname, shape):
    elevation = Image.open(fname)
    elevation = elevation.resize(shape, Image.NEAREST)
    elevation = np.transpose(np.array(elevation))
    elevation = elevation - np.min(elevation)
    return elevation

class BaseDataset(torch.utils.data.TensorDataset):
    """
    Base Dataset class that simply processes data and targets through optional transforms.

    Read more: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

    Parameters
    ----------
    data
        commonly these are torch tensors, numpy arrays, or PIL Images
    targets
        commonly these are torch tensors or numpy arrays
    transform
        function that takes a datum and returns the same
    target_transform
        function that takes a target and returns the same
    """

    def __init__(
        self,
        data: SequenceOrTensor,
        targets: SequenceOrTensor,
        shape: Tuple,
        transform: Callable = None,
        target_transform: Callable = None,
        return_eval: bool = False,
    ) -> None:
        if len(data) != len(targets):
            raise ValueError("Data and targets must be of equal length")
        super().__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        self.return_elevation = return_eval
        self.shape = shape

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Return a datum and its target, after processing by transforms.

        Parameters
        ----------
        index

        Returns
        -------
        (datum, target)
        """

        #datum, target = self.data[index], self.targets[index]
        datum = load_img(self.data[index],self.shape)
        target = load_mask(self.targets[index],self.shape)[:,:,0]
        elev_target = None
        if self.return_elevation:
            elev_target = load_elevation(self.targets[index].replace("label-chips", "eleva-chips"), self.shape)

        if self.transform is not None:
            datum, target, elev_target = self.transform(datum, target, elev_target)

        if self.return_elevation:
            return datum, target, elev_target

        return datum, target
