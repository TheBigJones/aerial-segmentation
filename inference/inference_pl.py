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

def chips_from_image(img, size=300, stride=1):
    shape = img.shape
    chips = []
    for x in range(0, shape[1], int(size/stride)):
        for y in range(0, shape[0], int(size/stride)):
            chip = img[y:y+size, x:x+size, :]
            y_pad = size - chip.shape[0]
            x_pad = size - chip.shape[1]
            chip = np.pad(chip, [(0, y_pad), (0, x_pad), (0, 0)], mode='constant')
            chips.append((chip, x, y))
    return chips

def run_inference_on_file(imagefile, predsfile, model, transform, size=300, batchsize=64, stride=2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model.model.to(device)
 
    with Image.open(imagefile).convert('RGB') as img:
        nimg = np.array(Image.open(imagefile).convert('RGB'))
        shape = nimg.shape
        chips = chips_from_image(nimg, size=size, stride=stride)
    
    num_classes = model.model.num_classes + 1 
    chips = [(chip, xi, yi) for chip, xi, yi in chips if chip.sum() > 0]
    prediction = np.zeros((num_classes, shape[0], shape[1]), dtype='uint8')
    #inp = transform(np.transpose(np.array([chip for chip, _, _ in chips]), (0, 3, 1, 2)), np.zeros((len(chips), size, size)))
    #print(transform(np.transpose(np.array(chips[0]), (0, 1, 2)), np.zeros((size, size)))[0])
    #print("#"*20)
    inp = torch.stack([transform(np.transpose(np.array(chip), (0, 1, 2)), np.zeros((size, size)))[0] for chip, _, _ in chips])
    # print("Prediction Shape: ",prediction.shape)
    # print("Input Shape: ",inp.shape)
    num_batches = (len(inp) + batchsize -1) // batchsize

    chip_preds_list = []
    for j in range(num_batches):
        batch_preds = model.predict(inp[j*batchsize : min((j+1)*batchsize, len(inp))].to(device))
        ignores_tensor = torch.zeros(batchsize, 1, inp.shape[2], inp.shape[3])
        batch_preds_with_ignore = torch.cat((ignores_tensor,batch_preds.to("cpu")), -3)
        print(batch_preds_with_ignore.shape)
        chip_preds_list.append(batch_preds_with_ignore)

    chip_preds = torch.cat(tuple(chip_preds_list))

    # print("Chipped Predictions Shape: ",chip_preds.shape)

    for (chip, x, y), pred in zip(chips, chip_preds):
        # print("X: ", x, "/Y: ", y)
        section = prediction[0, y:y+size, x:x+size].shape
        # print("Section Shape", section)
        prediction[:, y:y+size, x:x+size] = np.add(prediction[:, y:y+size, x:x+size], pred[:, :section[0], :section[1]])
    prediction = np.argmax(prediction, axis=-3)
    mask = category2mask(prediction)
    Image.fromarray(mask).save(predsfile)

def run_inference(dataset, model=None, basedir='predictions'):
    if not os.path.isdir(basedir):
        os.mkdir(basedir)
    if model is None:
        raise Exception("model is required")


    transforms_list = [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]
    transform = transforms.Compose(transforms_list)

    #for scene in train_ids + val_ids + test_ids:
    for scene in test_ids:
        imagefile = f'{dataset}/images/{scene}-ortho.tif'
        predsfile = os.path.join(basedir, f'{scene}-prediction.png')

        if not os.path.exists(imagefile):
            continue

        print(f'running inference on image {imagefile}.')
        print(f'saving prediction to {predsfile}.')
        run_inference_on_file(imagefile, predsfile, model, transform)
