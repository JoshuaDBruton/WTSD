import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import random
from utils.texture_loader import Texture
from PIL import Image
import cv2 as cv
from joblib import Parallel, delayed
from tqdm import tqdm
import skimage.measure as skm
from opensimplex import OpenSimplex
from skimage import exposure
import os

num_gen = 5000
DATA_PATH = "/home/joshua/Desktop/Work/simplex/data"
SAVE_PATH = os.path.join(DATA_PATH, "validation")
SAVE_INPUTS = os.path.join(SAVE_PATH, "inputs")
SAVE_TARGETS = os.path.join(SAVE_PATH, "targets")
HEIGHT = 256
WIDTH = 256
noise = True

textures = Texture(os.path.join(DATA_PATH, "textures"), 512, 512)


def noisy(image):
    image = cv.GaussianBlur(image, (3, 3), 0)
    row, col, ch = image.shape
    s_vs_p = 0.5
    amount = 0.001
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[tuple(coords)] = 1.0

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[tuple(coords)] = 0.0
    return out


def plains():
    tmp = OpenSimplex(int(random.random() * 10000))

    def noise(x, y):
        value = (tmp.noise2d(x/3.0, y/3.0) + 1) / 2.0

        value = value**0.25

        value = value - 0.6

        if value < 0:
            value = 0

        return value * 6.0

    return noise


def simp():
    tmp = OpenSimplex(int(random.random() * 10000))
    def noise(x, y):
        value = tmp.noise2d(x, y)
        return value

    return noise


def combined():
    simplex = plains()
    weights = plains()

    def noise(x, y):
        s = simplex(x, y)
        w = weights(x, y)
        return s*w

    return noise


def add_texture(tcanvas, binary_canvas):
    texture = textures()
    rot_scale = np.random.random_integers(0, 3)
    texture = np.array(Image.fromarray(texture).rotate(90*rot_scale))
    lower_y = random.randint(0,texture.shape[0]-HEIGHT)
    lower_x = random.randint(0,texture.shape[1]-WIDTH)
    texture = texture[lower_y:lower_y+HEIGHT, lower_x:lower_x+WIDTH, :]
    texture = Image.fromarray(texture)
    tcanvas = Image.composite(texture, Image.fromarray(tcanvas.astype(np.uint8)), Image.fromarray(np.uint8(255*binary_canvas)))
    return np.array(tcanvas)


def histogram_equalize(img):
    img_cdf, bin_centers = exposure.cumulative_distribution(img)
    return np.interp(img, bin_centers, img_cdf)


def generate_canvas(i):
    # Create empty 2d and 3d canvas
    canvas = np.zeros((HEIGHT, WIDTH))
    tcanvas = np.zeros((HEIGHT, WIDTH, 3))
    mask = np.zeros((HEIGHT, WIDTH))
    # Set k
    k = 4
    # Set x and y frequency scale
    freq_scale_x = random.uniform(30, 100)
    freq_scale_y = random.uniform(30, 100)

    model = combined()

    for x in range(HEIGHT):
        for y in range(WIDTH):
            canvas[x][y] = model(x/freq_scale_x, y/freq_scale_y)

    min = np.min(canvas)
    max = np.max(canvas)
    canvas = (canvas-min)/(max-min)

    canvas = histogram_equalize(canvas)

    actual = np.random.random_integers(2, k)

    bins = np.arange(0.0, 1.0+1.0/actual, 1.0/actual)
    possible_uppers = np.arange(0.0, 1.1, 0.1)
    # lower = 0
    # upper = random.choice(possible_uppers)

    for j in range(len(bins)-1):
        lower = bins[j]
        upper = bins[j+1]
        tcanvas = add_texture(tcanvas, np.where(np.logical_and(lower <= canvas, canvas <= upper), 1, 0))
        mask = np.where(np.logical_and(lower <= canvas, canvas <= upper), j, mask)
        # lower = upper
        # upper = 1.0
    canvas = tcanvas

    canvas = noisy(canvas)

    mask, _ = skm.label(mask, connectivity=2, return_num=True, background=-1)

    mask = np.array(mask, dtype=np.uint8)

    if len(np.unique(mask))>5:
        return generate_canvas(i)

    assert (len(canvas.shape) == 3), "canvas has more or less than 3 dimensions, it has shape {}".format(canvas.shape)
    assert (canvas.shape[2] == 3), "canvas is still not RGB, it has {} channels".format(canvas.shape[2])
    assert (canvas.shape[0] == mask.shape[0] and canvas.shape[1] == mask.shape[
        1]), "canvas and mask have different shapes in dim 0 or 1"
    assert (mask.dtype == np.uint8), "mask does not have the correct type, it is {}".format(mask.dtype)
    assert (canvas.dtype == np.uint8), "canvas does not have the correct type, it is {}".format(canvas.dtype)
    assert (np.max(mask) <= 7), "mask has some very high label values, like {}".format(np.max(mask))
    assert (len(mask.shape) == 2), "Mask seems to have to many dims, its shape is {}".format(mask.shape)

    plt.imsave(os.path.join(SAVE_INPUTS, "{}.jpg".format(i)), canvas)
    np.save(os.path.join(SAVE_TARGETS, "{}.npy".format(i)), mask)
    # plt.imsave(os.path.join(SAVE_TARGETS, "{}.png".format(i)), mask)


if __name__ == "__main__":
    _ = Parallel(n_jobs=-1)(delayed(generate_canvas)(i) for i in tqdm(range(num_gen)))
    # generate_canvas(0)
