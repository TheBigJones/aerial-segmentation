from PIL import Image
import numpy as np
import math
import os
import torch

from scipy import signal
from scipy.ndimage import interpolation

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

    for x in range(0, shape[1], size//stride):
        for y in range(0, shape[0], size//stride):
            chip = img[y:y+size, x:x+size, :]
            y_pad = size - chip.shape[0]
            x_pad = size - chip.shape[1]
            chip = np.pad(chip, [(0, y_pad), (0, x_pad), (0, 0)], mode='constant')
            chips.append((chip, x, y))
    return chips

def predict_on_chips(model, chips, size, shape, transform, batchsize = 16, smoothing = False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.model.to(device)

    model.set_image_size(size)

    num_classes = model.model.num_classes
    prediction = np.zeros((num_classes, shape[0], shape[1]))
    chips = [(chip, xi, yi) for chip, xi, yi in chips if chip.sum() > 0]
    # std chosen empirically
    if smoothing:
        smoothing_kernel = signal.gaussian(size, std=int(size/6)).reshape(size, 1)
        smoothing_kernel = np.outer(smoothing_kernel, smoothing_kernel)
    else:
        smoothing_kernel = np.ones((size, size))
    num_batches = (len(chips) + batchsize -1) // batchsize

    chip_preds_list = []
    for j in range(num_batches):
        # last batch can be smaller than batchsize
        size_batch = min((j+1)*batchsize, len(chips)) - j*batchsize
        batch_chips = chips[j*batchsize : min((j+1)*batchsize, len(chips))]
        inp = torch.stack([transform(np.transpose(np.array(chip), (0, 1, 2)), np.zeros((size, size)))[0] for chip, _, _ in batch_chips]).to(device)
        batch_preds = model.predict(inp)
        batch_preds = batch_preds.to("cpu")
        for (chip, x, y), pred in zip(batch_chips, batch_preds):
            section = prediction[0, y:y+size, x:x+size].shape
            prediction[:, y:y+size, x:x+size] = np.add(prediction[:, y:y+size, x:x+size], pred[:, :section[0], :section[1]]*smoothing_kernel[:section[0], :section[1]])

    return prediction

def run_inference_on_file(imagefile, predsfile, model, transform, size=300, batchsize=16, stride=1, smoothing = False):
    with Image.open(imagefile).convert('RGB') as img:
        nimg = np.array(Image.open(imagefile).convert('RGB'))
        shape = nimg.shape
        chips = chips_from_image(nimg, size=size, stride=stride)

    prediction = predict_on_chips(model, chips, size, shape, transform, batchsize = batchsize, smoothing = smoothing)

    ignore_mask = np.sum(prediction, axis =-3) > 0.0
    prediction = np.argmax(prediction, axis=-3)
    prediction[ignore_mask] += 1
    mask = category2mask(prediction)
    Image.fromarray(mask).save(predsfile)

def SDIV(x,y):
    return int((x+y-1)/y)

def valid_pixels(image):
    mask = np.sum(np.array(image.convert('RGB')), axis = -1) > 0.0
    return mask


def run_cascading_inference_on_file(imagefile, predsfile, model, transform, size=300, batchsize=16, stride=1, smoothing = False, alpha = 1./3, max_doubling_state=None, to_one_hot=True):
    num_classes = model.model.num_classes
    with Image.open(imagefile) as img:

        valid_pixel_mask = valid_pixels(img)

        original_shape = np.array(img.convert('RGB')).shape
        current_prediction = np.zeros((num_classes, original_shape[0], original_shape[1]))
        prediction_orig = np.zeros((num_classes, original_shape[0], original_shape[1]))

        larger_dim = max(original_shape[0], original_shape[1])
        num_doubling_states = int(math.ceil(np.log2(larger_dim / size)))

        if max_doubling_state is not None:
            num_doubling_states = min(num_doubling_states, max_doubling_state)

        for d in range(num_doubling_states, -1, -1):
            size_res_x = min(SDIV((larger_dim / 2**d), size)*size, original_shape[0])
            size_res_y = min(SDIV((larger_dim / 2**d), size)*size, original_shape[1])
            print(f"Predicting on image with resolution {size_res_x} x {size_res_y}")

            img_res = np.array(img.convert('RGB').resize((size_res_y, size_res_x)))

            shape = img_res.shape

            chips = chips_from_image(img_res, size=size, stride=stride)

            prediction = predict_on_chips(model, chips, size, shape, transform, batchsize = batchsize, smoothing = smoothing)

            # Factors to zoom to original shape. Mind that the original image is of shape (W, H, C), while the prediction is (num_classes, W, H)
            zoom_factors = [1, float(original_shape[0])/prediction.shape[1], float(original_shape[1])/prediction.shape[2]]

            # Interpolate array to get back to original shape
            interpolation.zoom(prediction,zoom_factors, output=prediction_orig, order=1)

            # Use old prediction weighted with alpha to predict new probabilites. To be formally correct, current_prediction would have to be
            # divided by (1+alpha) for proper normalization but as we use argmax anyway, this does not matter
            current_prediction = alpha * current_prediction + prediction_orig

            if to_one_hot:
                prediction_classes = np.argmax(current_prediction, axis=-3)
                # Cast to tensor to use PyTorch one_hot-function, cast back to numpy and transpose, because one-hot expands at the back of the array
                current_prediction = torch.nn.functional.one_hot(torch.from_numpy(prediction_classes), num_classes=num_classes).numpy().transpose(2,0,1)


        current_prediction = np.argmax(current_prediction, axis=-3)
        current_prediction[valid_pixel_mask] += 1
        invalid_pixel_mask = np.logical_not(valid_pixel_mask)
        current_prediction[invalid_pixel_mask] = 0
        mask = category2mask(current_prediction)
        Image.fromarray(mask).resize((original_shape[1], original_shape[0]), resample=Image.NEAREST).save(predsfile)

def run_inference(dataset, model=None, basedir='predictions', stride=1, smoothing=False, size=300):
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
        run_cascading_inference_on_file(imagefile, predsfile, model, transform, stride=stride, smoothing=smoothing, size=size)
