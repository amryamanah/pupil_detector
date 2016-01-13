import os
import logging
from time import time
import pprint

import cv2
import numpy as np
from .utils import get_chroma_value
logger = logging.getLogger(__name__)


class BaseClassifier:
    def __init__(self):
        self.dataset = None
        self.dataset_label = None
        self.train_dataset = None
        self.train_dataset_label = None
        self.test_dataset = None
        self.test_dataset_label = None

    def load_dataset(self):
        pass

    def training(self):
        pass

    def _bg_img(self, im_path):
        image = cv2.imread(im_path)
        b, g, r = cv2.split(image)
        r = np.zeros(b.shape, dtype=np.uint8)
        bgr = cv2.merge([b, g, r])
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return gray


