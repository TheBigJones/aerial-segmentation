from PIL import Image
import numpy as np
import math
import os
import torch

from data.config import train_ids, test_ids, val_ids, LABELMAP_RGB
from data import transforms

def category2mask(img):
    """ Convert a category image to color mask """
    if len(img) == 3:
        if img.shape[2] == 3:
            img = img[:, :, 0]

    mask = np.zeros(img.shape[:2] + (3, ), dtype='uint8')

    for category, mask_color in LABELMAP_RGB.items():
        locs = np.where(img == category)
        mask[locs] = mask_color

    return mask

def chips_from_image(img, size=300):
    shape = img.shape

    chip_count = math.ceil(shape[1] / size) * math.ceil(shape[0] / size)

    chips = []
    for x in range(0, shape[1], size):
        for y in range(0, shape[0], size):
            chip = img[y:y+size, x:x+size, :]
            y_pad = size - chip.shape[0]
            x_pad = size - chip.shape[1]
            chip = np.pad(chip, [(0, y_pad), (0, x_pad), (0, 0)], mode='constant')
            chips.append((chip, x, y))
    return chips

def run_inference_on_file(imagefile, predsfile, model, transform, size=300):
    with Image.open(imagefile).convert('RGB') as img:
        nimg = np.array(Image.open(imagefile).convert('RGB'))
        shape = nimg.shape
        chips = chips_from_image(nimg)

    chips = [(chip, xi, yi) for chip, xi, yi in chips if chip.sum() > 0]
    prediction = np.zeros(shape[:2], dtype='uint8')
    #inp = transform(np.transpose(np.array([chip for chip, _, _ in chips]), (0, 3, 1, 2)), np.zeros((len(chips), size, size)))
    #print(transform(np.transpose(np.array(chips[0]), (0, 1, 2)), np.zeros((size, size)))[0])
    #print("#"*20)
    inp = torch.stack([transform(np.transpose(np.array(chip), (0, 1, 2)), np.zeros((size, size)))[0] for chip, _, _ in chips])

    print(inp.shape)

    chip_preds = model.predict(inp)

    print(chip_preds.shape)

    for (chip, x, y), pred in zip(chips, chip_preds):
        category_chip = np.argmax(pred, axis=-3) + 1
        section = prediction[y:y+size, x:x+size].shape
        prediction[y:y+size, x:x+size] = category_chip[:section[0], :section[1]]

    mask = category2mask(prediction)
    Image.fromarray(mask).save(predsfile)

def run_inference(dataset, model=None, basedir='predictions'):
    if not os.path.isdir(basedir):
        os.mkdir(basedir)
    if model is None:
        raise Exception("model is required")


    transforms_list = [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]
    transform = transforms.Compose(transforms_list)

    for scene in train_ids + val_ids + test_ids:
        imagefile = f'{dataset}/images/{scene}-ortho.tif'
        predsfile = os.path.join(basedir, f'{scene}-prediction.png')

        if not os.path.exists(imagefile):
            continue

        print(f'running inference on image {imagefile}.')
        print(f'saving prediction to {predsfile}.')
        run_inference_on_file(imagefile, predsfile, model, transform)
