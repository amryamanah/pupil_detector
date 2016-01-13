# -- coding: utf-8 --

__author__ = 'amryfitra'
from skimage import feature
import numpy as np
from IPython import embed


class LBPDescriptor:
    def __init__(self, num_points, radius):
        self.num_points = num_points
        self.radius = radius

    def describe(self, gray, eps=1e-7):
        lbp = feature.local_binary_pattern(gray, self.num_points, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=range(0, self.num_points + 2),
                                 range=(0, self.num_points + 1))
        return hist




