""" Props to the dudes from https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py """

import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, elev_target=None):
        for t in self.transforms:
            image, target, elev_target = t(image, target, elev_target)
        return image, target, elev_target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target, elev_target=None):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=Image.NEAREST)
        if elev_target is not None:
            elev_target = F.resize(elev_target, size, interpolation=Image.NEAREST)
        return image, target, elev_target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target, elev_target=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
            if elev_target is not None:
                elev_target = F.hflip(elev_target)
        return image, target, elev_target

class RandomVerticalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target, elev_target=None):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
            if elev_target is not None:
                elev_target = F.vflip(elev_target)
        return image, target, elev_target

class RandomRotation(object):
    def __init__(self, degrees=90):
        self.degrees = degrees
        # Subtract one to avoid having rotation by zero degrees twice
        self.num_rotations = int(360//degrees)-1

    def __call__(self, image, target, elev_target=None):
        degrees = random.randint(0, self.num_rotations) * self.degrees

        image = F.rotate(image, self.degrees)
        target = F.rotate(target, self.degrees)
        if elev_target is not None:
            elev_target = F.rotate(elev_target, self.degrees)
        return image, target, elev_target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, elev_target=None):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        if elev_target is not None:
            elev_target = pad_if_smaller(elev_target, self.size, fill=255)
            elev_target = F.crop(elev_target, *crop_params)
        return image, target, elev_target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, elev_target=None):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        if elev_target is not None:
            elev_target = F.center_crop(elev_target, self.size)
        return image, target, elev_target


class ToTensor(object):
    def __call__(self, image, target, elev_target=None):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        if elev_target is not None:
            elev_target = torch.as_tensor(np.array(elev_target), dtype=torch.float32)
        return image, target, elev_target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target, elev_target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if elev_target is not None:
            elev_target = elev_target/255.
        return image, target, elev_target
