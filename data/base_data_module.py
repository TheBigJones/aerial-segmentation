"""Base DataModule class."""
from pathlib import Path
from typing import Collection, Dict, Optional, Tuple, Union
import argparse

from torch.utils.data import ConcatDataset, DataLoader
import pytorch_lightning as pl

from data.util_torch import BaseDataset
import torch

import numpy as np

def load_and_print_info(data_module_class) -> None:
    """Load EMNISTLines and print info."""
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args = parser.parse_args()
    dataset = data_module_class(args)
    dataset.prepare_data()
    dataset.setup()
    print(dataset.data_train[0])


BATCH_SIZE = 8
NUM_WORKERS = -1
IMAGE_SIZE=300
READ_CONFIG=True


class ParserError(Exception):
    pass

class BaseDataModule(pl.LightningDataModule):
    """
    Base DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)

        self.read_config = self.args.get("read_config", READ_CONFIG)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)
        if self.num_workers >=0 and self.read_config:
            raise ParserError("read_config and num_workers-arg set at the same time.")
        elif self.read_config:
            conf = self.read_config_from_file()
            print(conf)
            self.num_workers = int(conf["num_workers"])
        elif self.num_workers < 0:
            raise ParserError("Must specify number of workers explicitely via read_config or num_workers.")


        self.image_size = self.args.get("image_size", IMAGE_SIZE)

        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))

        # Make sure to set the variables below in subclasses
        self.dims: Tuple[int, ...]
        self.mapping: Collection
        self.data_train: Union[BaseDataset, ConcatDataset]
        self.data_val: Union[BaseDataset, ConcatDataset]
        self.data_test: Union[BaseDataset, ConcatDataset]
        self.class_labels: Dict

    def read_config_from_file(self):
        conf = {}
        for line in open("config.dat"):
            if line.startswith("#"):
                continue
            arg,val = line.split("=")
            conf[arg.strip()] = val.strip()
        return conf

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size", type=int, default=BATCH_SIZE, help="Number of examples to operate on per forward step."
        )
        parser.add_argument(
            "--num_workers", type=int, default=NUM_WORKERS, help="Number of additional processes to load data."
        )
        parser.add_argument(
            "--image_size", type=int, default=IMAGE_SIZE, help="Size of input images to resize to."
        )
        parser.add_argument(
            "--read_config", action="store_true", default=False
        )
        return parser

    def config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""
        return {"input_dims": self.dims, "mapping": self.mapping, "class_labels": self.class_labels}

    def prepare_data(self, *args, **kwargs) -> None:
        """
        Use this method to do things that might write to disk or that need to be done only from a single GPU
        in distributed settings (so don't set state `self.x = y`).
        """

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Split into train, val, test, and set dims.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """

    # Probs to the dudes from https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
    def worker_init_fn(self, worker_id):
        np.random.seed(torch.randint(high=2**32 - 1, size=(1,))[0] + worker_id)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
            worker_init_fn=self.worker_init_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
            worker_init_fn=self.worker_init_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
            worker_init_fn=self.worker_init_fn
        )
