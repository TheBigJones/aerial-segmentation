"""AerialData DataModule"""
import argparse

from torch.utils.data import random_split
from torchvision import transforms

from base_data_module import BaseDataModule, load_and_print_info

import os


ELEVATION = False
DATASET = "dataset-sample"

def load_lines(fname):
    with open(fname, 'r') as f:
        return [l.strip() for l in f.readlines()]

class AerialData(BaseDataModule):
    """
    AerialData DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.dims = (1, 28, 28)  # dims are returned when calling `.size()` on this object.
        self.output_dims = (1,)
        self.mapping = list(range(10))
        self.elevation = self.args.get("elevation", ELEVATION)
        self.dataset = self.args.get("dataset", DATASET)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--elevation", type=bool, default=ELEVATION, help="Flag to indicate whether to use also elevation as target."
        )
        parser.add_argument(
            "--dataset", type=str, default=DATASET, help="Datset to use. Choose from 'dataset-sample' and 'dataset-medium'."
        )
        return parser

    def prepare_data(self, *args, **kwargs) -> None:
        pass

    def setup(self, stage=None) -> None:
        """Split into train, val, test, and set dims."""
        train_imgs = [f'../{self.dataset}/image-chips/{fname}' for fname in load_lines(f'../{self.dataset}/train.txt')]
        valid_imgs = [f'../{self.dataset}/image-chips/{fname}' for fname in load_lines(f'../{self.dataset}/valid.txt')]
        test_imgs = [f'../{self.dataset}/image-chips/{fname}' for fname in load_lines(f'../{self.dataset}/test.txt')]

        train_labels = [t.replace("image-chips", "label-chips") for t in train_imgs]
        valid_labels = [v.replace("image-chips", "label-chips") for v in valid_imgs]
        test_labels = [t.replace("image-chips", "label-chips") for t in test_imgs]

        ## TODO: check whether format is correct: Should it be [[datum, targt]] or [[datum], [target]]
        self.data_train = [e for e in zip(train_imgs, train_labels)]
        self.data_valid = [e for e in zip(valid_imgs, valid_labels)]
        self.data_test = [e for e in zip(test_imgs, test_labels)]


if __name__ == "__main__":
    load_and_print_info(AerialData)
