import cmath
from math import atan2
from random import random
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
from scipy.spatial import Delaunay
from tqdm import tqdm

def in_hull(p, hull):
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0

def dft(xs):
    pi = 3.14
    return [sum(x * cmath.exp(2j*pi*i*k/len(xs))
                for i, x in enumerate(xs))
            for k in range(len(xs))]

def interpolateSmoothly(xs, N):
    """For each point, add N points."""
    fs = dft(xs)
    half = (len(xs) + 1) // 2
    fs2 = fs[:half] + [0]*(len(fs)*N) + fs[half:]
    return [x.real / len(xs) for x in dft(fs2)[::-1]]

class BlobMaker:
    def __init__(self, width, height, smoothness=100, x_min=0.1, x_max=0.9, y_min=0.1, y_max=1.0):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.width = width
        self.height = height
        self.smoothness = smoothness

    def generate_blob(self):
        xs = np.random.uniform(low=self.x_min, high=self.x_max, size=(self.smoothness,))
        ys = np.random.uniform(low=self.y_min, high=self.y_max, size=(self.smoothness,))
        pts = [v for v in zip(ys, xs)]

        hull = ConvexHull(pts)
        vertices = [pts[v] for v in hull.vertices]

        xs, ys = [interpolateSmoothly(zs, self.smoothness) for zs in zip(*vertices)]
        xs = np.array(xs)
        ys = np.array(ys)
        xs = np.array(xs*self.height, dtype=np.int)
        ys = np.array(ys*self.width, dtype=np.int)
        vs = [v for v in zip(xs, ys)]
        d_hull = Delaunay(vs)

        mask = np.zeros((self.height,self.width))
        for y in range(self.height):
            for x in range(self.width):
                if in_hull([y, x], d_hull):
                    mask[y][x] = 1
                else:
                    mask[y][x] = 0

        return mask

if __name__=="__main__":
    bg = BlobMaker(256, 256)
    bg.generate_blob()
