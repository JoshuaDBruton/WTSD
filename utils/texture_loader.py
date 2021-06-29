import cv2 as cv
import os
import glob
import numpy as np
import random
import matplotlib.pyplot as plt

class Texture:
    def __init__(self, texture_dir, height, width):
        self.texture_dir = texture_dir
        self.texture_files = os.listdir(texture_dir)
        self.height = height
        self.width = width

    def __call__(self):
        texture = None
        while texture is None:
            try:
                f = random.choice(self.texture_files)
                texture = cv.imread(os.path.join(self.texture_dir, f))
                texture = cv.resize(texture, (self.width, self.height), interpolation = cv.INTER_AREA)
            except:
                 texture = None
        return texture[:, :, ::-1]



if __name__ == "__main__":
    textures = Texture("/home/joshua/Desktop/Work/simplex/data/textures", 512, 512)
    textures()
