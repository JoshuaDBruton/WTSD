import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import random
from utils.texture_loader import texload
from PIL import Image
import cv2 as cv
from joblib import Parallel, delayed
from tqdm import tqdm
import skimage.measure as skm
from opensimplex import OpenSimplex
from skimage import exposure
from blob_gen import BlobMaker
from scipy.ndimage import gaussian_filter
import os


def simple_curve(value):
    start = 0.4
    end = 0.6
    if value < start:
        return 0.0
    if value > end:
        return 1.0
    return (value - start) * (1 / (end - start))


def interpolate(a, b, weight):
    new_weight = simple_curve(weight)

    return a * (1 - new_weight) + b * new_weight


def simple_scurve():
    tmp = OpenSimplex(int(random.random() * 10000))

    def noise(x, y):
        noise = (tmp.noise2d(x/5.0, y/5.0) + 1) / 2.0

        return interpolate(0.0, 1.0, noise) * 10.0

    return noise


def plains():
    tmp = OpenSimplex(int(random.random() * 10000))

    def noise(x, y):
        value = (tmp.noise2d(x/random.randint(1, 11), y/random.randint(1,11)) + 1)

        value = value**0.25

        value = value - 0.6

        if value < 0:
            value = 0

        return value * 6.0

    return noise


def mountains():
    tmp = OpenSimplex(int(random.random() * 10000))

    def noise(x, y):
        value = (tmp.noise2d(x*2.0, y) + 1) / 2.0

        value = value

        return value

    return noise


def combined():
    m_values = mountains()
    p_values = plains()
    weights = simple_scurve()

    def noise(x, y):
        m = m_values(x, y)
        p = p_values(x, y)
        w = weights(x, y) / 10.0
        return (p * w) + (m * (1 - w))
        return (p * w)

    return noise


def other_com():
    p_values = plains()
    w_values = plains()

    def noise(x, y):
        s = p_values(x, y)
        w = w_values(x, y)
        return s*w

    return noise


class SilMaker:
    def __init__(self, n, width, height, data_path):
        self.n = n
        self.blob_maker = BlobMaker(width=width, height=height)
        self.width = width
        self.height = height
        self.data_path = data_path

    def generate_canvas(self, i):
        mask = self.blob_maker.generate_blob()

        thresh = np.random.uniform(0, 1)
        if thresh>=0.5:
            mask = 1 - mask

        m1 = gaussian_filter(mask, sigma=np.random.random_integers(2,10))

        simplex = combined()
        simplex2 = plains()
        m2 = np.zeros((self.height, self.width))
        m3 = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                m2[y][x] = simplex(x, y)
                m3[y][x] = simplex2(x, y)

        m = m1*m2

        mask = 1 - mask

        m1 = gaussian_filter(mask, sigma=np.random.random_integers(2,10))

        m += m1*m3

        m = (m-np.min(m))/(np.max(m)-np.min(m))
        m = np.array(m, dtype=np.float32)

        if len(m.shape) == 2:
            m = cv.cvtColor(m, cv.COLOR_GRAY2RGB)

        m = np.array(m*255, dtype=np.uint8)
        mask = np.where(mask>0, 1, 0)
        mask = np.array(mask, dtype=np.uint8)

        assert (len(m.shape) == 3), "m has more or less than 3 dimensions, it has shape {}".format(m.shape)
        assert (m.shape[2] == 3), "m is still not RGB, it has {} channels".format(m.shape[2])
        assert (m.shape[0] == mask.shape[0] and m.shape[1] == mask.shape[1]), "m and mask have different shapes in dim 0 or 1"
        assert (mask.dtype == np.uint8), "y does not have the correct type, it is {}".format(mask.dtype)
        assert (m.dtype == np.uint8), "x does not have the correct type, it is {}".format(m.dtype)

        plt.imsave(os.path.join(self.data_path, "inputs/"+str(i)+".jpg"), m)
        # plt.imsave(os.path.join(self.data_path, "targets/"+str(i)+".png"), mask)
        np.save(os.path.join(self.data_path, "targets/"+str(i)+".npy"), mask)

    def generate(self):
        _ = Parallel(n_jobs=-1)(delayed(self.generate_canvas)(i) for i in tqdm(range(self.n)))


if __name__=="__main__":
    # _ = Parallel(n_jobs=-1)(delayed(generate_canvas)(i) for i in tqdm(range(num_gen)))
    # generate_canvas(0)
    generator = SilMaker(n=400, width=256, height=256, data_path="/home/joshua/Desktop/Work/simplex/data/validation")
    generator.generate()
