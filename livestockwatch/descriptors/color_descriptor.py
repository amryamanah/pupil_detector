import numpy as np
import cv2


class ColorDescriptor:
    def __init__(self, bins, color_space):
        self.bins = bins
        self.color_space = color_space

    def describe(self, image):
        if self.color_space == "hsv":
            hist = cv2.calcHist([image], [0, 1, 2], None, self.bins, [0, 180, 0, 256, 0, 256])
        elif self.color_space == "rgb":
            hist = cv2.calcHist([image], [0, 1, 2], None, self.bins, [0, 256, 0, 256, 0, 256])
        else:
            raise "Unsupported color space"

        cv2.normalize(hist, hist)
        hist = hist.flatten()
        return hist
