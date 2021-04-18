"""AerialData DataModule"""
import argparse

from torch.utils.data import random_split

import data.transforms as transforms

from data.util_torch import BaseDataset

from data.base_data_module import BaseDataModule, load_and_print_info

import os

from data.config import LABELS

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
        #TODO normalize equal to pretrained model
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                           std=[0.229, 0.224, 0.225])])
        self.dims = (3, self.image_size, self.image_size)  # dims are returned when calling `.size()` on this object.
        # create dict {0:class1, 1:class2 ....}
        self.class_labels = dict(zip(range(len(LABELS)),LABELS))

        self.mapping = (len(self.class_labels), self.image_size, self.image_size)
        self.elevation = self.args.get("elevation", ELEVATION)
        self.dataset = self.args.get("dataset", DATASET)



    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
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
        #TODO work on project folder level
        train_imgs = [f'./{self.dataset}/image-chips/{fname}' for fname in load_lines(f'./{self.dataset}/train.txt')]
        valid_imgs = [f'./{self.dataset}/image-chips/{fname}' for fname in load_lines(f'./{self.dataset}/valid.txt')]
        test_imgs = [f'./{self.dataset}/image-chips/{fname}' for fname in load_lines(f'./{self.dataset}/test.txt')]

        train_labels = [t.replace("image-chips", "label-chips") for t in train_imgs]
        valid_labels = [v.replace("image-chips", "label-chips") for v in valid_imgs]
        test_labels = [t.replace("image-chips", "label-chips") for t in test_imgs]

        ## TODO: check whether format is correct: Should it be [[datum, targt]] or [[datum], [target]]
        self.data_train = BaseDataset([e for e in train_imgs], [e for e in train_labels], shape=(self.image_size,self.image_size), transform = self.transform)
        self.data_val = BaseDataset([e for e in valid_imgs], [e for e in valid_labels], shape=(self.image_size,self.image_size), transform = self.transform)
        self.data_test = BaseDataset([e for e in test_imgs], [e for e in test_labels], shape=(self.image_size,self.image_size))


if __name__ == "__main__":
    load_and_print_info(AerialData)