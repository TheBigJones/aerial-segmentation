"""Base Dataset class."""
from typing import Any, Callable, Dict, Sequence, Tuple, Union
import torch
from PIL import Image
import numpy as np
import random

SequenceOrTensor = Union[Sequence, torch.Tensor]


def load_img(fname):
    return np.array(Image.open(fname))


class BaseDataset(torch.utils.data.Dataset):
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
        self.target_transform = target_transform
        self.return_elevation = return_eval

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
        datum = load_img(self.data[index])
        target = load_img(self.targets[index])

        if self.return_elevation:
            target = [target, load_img(self.targets[index].replace("label-chips", "eleva-chips"))]

        # Adjusted trasnformations to follow https://github.com/pytorch/vision/issues/9#issuecomment-304224800
        # This is necessary to allow for the same transformations of target and datum
        seed = np.random.randint(2147483647)
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.transform is not None:
            datum = self.transform(datum)

        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.target_transform is not None:
            if self.return_elevation:
                target = [self.target_transform(target[0]), self.target_transform(target[1])]
            else:
                target = self.target_transform(target)

        return datum, target
