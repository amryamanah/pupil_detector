import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage import color, io
from skimage.segmentation import clear_border
from skimage.measure import regionprops, label
import math


class PupilAnalysisException(Exception):
    """Base class for Pupil Analysis"""
    pass

class NoPupilDetected(PupilAnalysisException):
    """Raised when no pupil is detected"""
    pass

class ImpartialPupilDetected(PupilAnalysisException):
    """Raised when impartial pupil is detected"""
    pass


class Parameter:
    def __init__(self, hue_max, hue_min, sat_max=1.0, sat_min=0.0, val_max=1.0, val_min=0.0):
        self.hue_max = hue_max
        self.hue_min = hue_min
        self.sat_max = sat_max
        self.sat_min = sat_min
        self.val_max = val_max
        self.val_min = val_min


def create_mask(RGB, parameter):

    # Convert RGB image to chosen color space
    hsv_img = color.rgb2hsv(RGB)

    hue_min = parameter.hue_min
    hue_max = parameter.hue_max
    sat_min = parameter.sat_min
    sat_max = parameter.sat_max
    val_min = parameter.val_min
    val_max = parameter.val_max

    h = np.logical_and(
        (hsv_img[:, :, 0] >= hue_min),
        (hsv_img[:, :, 0] <= hue_max))
    s = np.logical_and(
        (hsv_img[:, :, 1] >= sat_min),
        (hsv_img[:, :, 1] <= sat_max))
    v = np.logical_and(
        (hsv_img[:, :, 2] >= val_min),
        (hsv_img[:,:,2] <= val_max))

    BW = np.logical_and(h, np.logical_and(s,v))

    return BW


def detect_pupil(rgb, parameter):

    bw = create_mask(rgb, parameter)

    cleared = bw.copy()
    clear_border(cleared)

    # label image regions
    label_image = label(cleared)
    borders = np.logical_xor(bw, cleared)
    label_image[borders] = -1

    props = None
    pupil_area = 0

    s = regionprops(label_image)
    if s:
        for region in s:

            if props is None:
                props = region
            else:
                if props.area < region.area:
                    props = region

        boundary = [0 for _ in range(4)]
        boundary[0], boundary[1], boundary[2], boundary[3] = props.bbox

        for x in boundary:
            if x <= 1:
                props = None

        if props:
            pupil_area = (props.major_axis_length/2) * (props.minor_axis_length/2) * np.pi
            # print(props.bbox)

        if not props.area and props.area < 5000:
            pupil_area = 0
            props = None
            print("Area = {} is less than 5000 px".format(props.area))
        if pupil_area == 0:
            props = None
    else:
        props = None

    return props, pupil_area


def save_pupil_analysis_result(rgb, props, result_path):

    fig, ax = plt.subplots()
    ax.axis("off")
    ax.set_title("Area=%s" % props.area)
    ax.imshow(rgb)

    y0, x0 = props.centroid
    orientation = props.orientation
    x1 = x0 + math.cos(orientation) * 0.5 * props.major_axis_length
    y1 = y0 - math.sin(orientation) * 0.5 * props.major_axis_length
    x2 = x0 - math.sin(orientation) * 0.5 * props.minor_axis_length
    y2 = y0 - math.cos(orientation) * 0.5 * props.minor_axis_length

    ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
    ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
    ax.plot(x0, y0, '.g', markersize=15)

    minr, minc, maxr, maxc = props.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-b', linewidth=2.5)
    # print("attempt to save analysis result to : {}".format(result_path))
    fig.savefig(result_path, bbox_inches='tight')
    plt.close(fig)

class EllipseAnalysis(object):

    def __init__(self, major_axis, minor_axis):
        self.semi_major_axis = major_axis/2
        self.semi_minor_axis = minor_axis/2
        self.area = self.calculate_area()
        self.perimeter = self.calculate_perimeter()
        self.eccentricity = self.calculate_eccentricity()
        self.ipr = self.calculate_ipr()

    def calculate_area(self):
        if self.semi_major_axis == 0:
            return 0
        else:
            area = np.pi * self.semi_major_axis * self.semi_minor_axis
            return area

    def calculate_perimeter(self):
        if self.semi_major_axis == 0:
            return 0
        else:
            perimeter = (2.00 * np.pi) * math.sqrt(((self.semi_major_axis**2)+(self.semi_minor_axis**2))/2.00)
            return perimeter

    def calculate_eccentricity(self):
        if self.semi_major_axis == 0:
            return 0
        else:
            eccentricity = math.sqrt((self.semi_major_axis**2) - (self.semi_minor_axis**2)) / self.semi_major_axis
            return eccentricity

    def calculate_ipr(self):
        if self.perimeter == 0:
            return 0
        else:
            ipr = (4 * np.pi * self.area) / (self.perimeter**2)
            return ipr


def ellipse_normalized_area(area, max_area):
    if max_area == 0:
        return 0
    else:
        return area/max_area


def ellipse_calculate_ca(area, max_area):
    if max_area == 0:
        return 0
    else:
        ca = (max_area - area)/max_area
        return ca