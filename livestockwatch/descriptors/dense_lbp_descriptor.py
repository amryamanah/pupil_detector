# -- coding: utf-8 --

__author__ = 'amryfitra'
from skimage import feature
import numpy as np
from .lbp_descriptor import LBPDescriptor
from IPython import embed
import cv2


class DenseLBPDescriptor:
    POINT_SEED = [
        [0, 31], [15, 47], [31, 63], [47, 79],
        [63, 95], [79, 111], [95, 127], [111, 143],
        [127, 159], [143, 175], [159, 191], [175, 207],
        [191, 223], [207, 239], [223, 255], [239, 271],
        [255, 287], [271, 303], [287, 319]
    ]

    def __init__(self, num_points, radius):
        self.num_points = num_points
        self.radius = radius
        self.lbp_descriptor = LBPDescriptor(self.num_points, self.radius)

    def _gen_convol(self):
        return [[x0, y0] for x0 in DenseLBPDescriptor.POINT_SEED for y0 in DenseLBPDescriptor.POINT_SEED]

    def describe(self, gray, eps=1e-7):
        lst_hist = []
        lst_convol = self._gen_convol()

        # gray = cv2.GaussianBlur(gray, ())

        for x_point, y_point in lst_convol:
            a = None
            if 0 in x_point:
                xend = x_point[1] + 1
            else:
                xend = x_point[1]

            if 0 in y_point:
                yend = y_point[1] + 1
            else:
                yend = y_point[1]

            patch = gray[y_point[0]:yend, x_point[0]:xend]
            if a:
                break
            # print("ypoint = {}, xpoint = {}, patch shape = {}".format(y_point, x_point, patch.shape))
            hist = self.lbp_descriptor.describe(patch)

            # # normalize the histogram
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)
            lst_hist.append(hist)

        lst_hist = np.array(lst_hist)

        return lst_hist




