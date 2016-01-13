# -- coding: utf-8 --

__author__ = 'amryfitra'

import csv
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import color, io
from IPython import embed


def is_too_dark(image, val_lower=35.0):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    means, stds = cv2.meanStdDev(grayscale)
    if means.sum() < val_lower:
        return True
    else:
        return False


def equ_hist_color_image(image):
    b, g, r = cv2.split(image)
    equ_b = cv2.equalizeHist(b)
    equ_g = cv2.equalizeHist(g)
    equ_r = cv2.equalizeHist(r)
    return cv2.merge([equ_b, equ_g, equ_r])


def get_list_of_frame(video_path, step, max_frametime, pupil_finder, nopl_result_path):
    video = cv2.VideoCapture(video_path)
    first_eye_timestamp = None
    lst_frame = []
    while True:
        (grabbed, frame) = video.read()

        if not grabbed:
            break

        frame_timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.00
        frame_timestamp = float("{:.2f}".format(frame_timestamp))
        if not first_eye_timestamp and frame_timestamp > 0.5:
            final_eye_flag, final_point, final_patch_equ = pupil_finder.detect_pupil(
                    frame, nopl_result_path,
                    frame_timestamp)
            if final_eye_flag:
                print("Set First frame timestamp at {}s".format(frame_timestamp))
                first_eye_timestamp = frame_timestamp
                for x in np.arange(first_eye_timestamp, max_frametime, step):
                    lst_frame.append(x)

    video.release()
    return lst_frame


def is_skipped_folder(dirpath, skip_folder):
    part_dirpath = dirpath.split(os.sep)
    for x in part_dirpath:
        if x in skip_folder:
            return True
    return False


class Rectangle:
    def __init__(self, point):
        self.right = max([point[0], point[2]])
        self.left = min([point[0], point[2]])
        self.top = max([point[1], point[3]])
        self.bottom = min([point[1], point[3]])


def measure_overlap_ratio(pred_rect, real_rect, win_size):
    if pred_rect.right <= real_rect.left or real_rect.right <= pred_rect.left or \
                    pred_rect.top <= real_rect.bottom or real_rect.top <= pred_rect.bottom:
        # There is no intersection in these cases
        return False, 0
    else:
        if pred_rect.right >= real_rect.right and pred_rect.left <= real_rect.left:
            overlap_width = real_rect.right - real_rect.left
        elif real_rect.right >= pred_rect.right and real_rect.left <= pred_rect.left:
            overlap_width = pred_rect.right - pred_rect.left
        elif pred_rect.right >= real_rect.right:
            overlap_width = real_rect.right - pred_rect.left
        else:
            overlap_width = pred_rect.right - real_rect.left

        if pred_rect.top >= real_rect.top and pred_rect.bottom <= real_rect.bottom:
            overlap_height = real_rect.top - real_rect.bottom
        elif real_rect.top >= pred_rect.top and real_rect.bottom <= pred_rect.bottom:
            overlap_height = pred_rect.top - pred_rect.bottom
        elif pred_rect.top >= real_rect.top:
            overlap_height = real_rect.top - pred_rect.bottom
        else:
            overlap_height = pred_rect.top - real_rect.bottom

        area_overlap = overlap_height * overlap_width
        area_bbox = win_size * win_size
        area_union = (2*area_bbox) - area_overlap
        overlap_ratio = area_overlap / area_union

    return True, overlap_ratio


def get_chroma_value(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    chroma = np.sqrt((a*a) + (b*b))

    return chroma


def get_blue_ratio(image):
    b, g, r = cv2.split(image)
    sum_channel = b + g + r
    ratio = (b / sum_channel) * 255

    return ratio.astype("uint8")


def is_mask(bgr):
    b, g, r = cv2.split(bgr)
    flag = np.array_equiv(b,g) and np.array_equiv(b, r)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray, flag


def getfiles(dirpath):
    a = [os.path.join(dirpath, s) for s in os.listdir(dirpath)
         if s.endswith(".bmp")]
    a.sort(key=lambda s: get_timestamp(s))
    return a


def get_timestamp(imgpath):
    filename = os.path.basename(imgpath)
    img_number = int(re.findall(r'\d+', filename)[0])
    if img_number < 11:
        timestamp = (100 * img_number)/1000.00
    else:
        timestamp = (200 * img_number)/1000.00
        timestamp = timestamp - 1
    return timestamp


def get_nopl_distance(imagelog_csv, imgpath):
    filename = os.path.basename(imgpath)
    img_number = int(re.findall(r'\d+', filename)[0])
    for imagelog in imagelog_csv:
        if imagelog[0].isdigit():
            if int(imagelog[0]) == img_number:
                return float(imagelog[2])


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def write_csv_result(csv_path, header, data):
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()

    with open(csv_path, "a") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writerow(data)


def plot_confusion_matrix(result_folder, cm, filename, title='Confusion matrix'):
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(["negative", "positive"]))
    plt.xticks(tick_marks, ["negative", "positive"], rotation=45)
    plt.yticks(tick_marks, ["negative", "positive"])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(result_folder, "{}.png".format(filename)))
    plt.close()


def special_max_r_gb_filter(image):
    # split the image into its BGR components
    (B, G, R) = cv2.split(image)

    # find the maximum pixel intensity values for each
    # (x, y)-coordinate,, then set all pixel values less
    # than M to zero
    M = np.maximum(np.maximum(R, G), B)
    #M = np.maximum(G, B)
    R[R < M] = 0
    G[G < M] = 0
    B[B < M] = 0

    green_frame = G.nonzero()
    green_coordinates = [(k, v) for k, v in zip(green_frame[0], green_frame[1])]

    median_blue = np.nanmean(B)
    # median_blue = 255
    for k, v in green_coordinates:
        if B[k,v] == R[k,v] == G[k,v]:
            G[k, v] = 0
            R[k, v] = 0
            B[k, v] = median_blue
        else:
            B[k, v] = G[k, v]
            G[k, v] = 0

    # merge the channels back together and return the image
    return cv2.merge([B, G, R])


def max_rgb_filter(image):
    # split the image into its BGR components
    (B, G, R) = cv2.split(image)

    # find the maximum pixel intensity values for each
    # (x, y)-coordinate,, then set all pixel values less
    # than M to zero
    M = np.maximum(np.maximum(R, G), B)
    R[R < M] = 0
    G[G < M] = 0
    B[B < M] = 0

    # merge the channels back together and return the image
    return cv2.merge([B, G, R])

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


def write_as_png(newpath, frame):
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    cv2.imwrite(newpath, frame, png_params)


def create_mask(frame):

    # Convert RGB image to chosen color space
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv_img = color.rgb2hsv(RGB)

    hue_min = 0.389
    hue_max = 0.737
    sat_min = 0.0
    sat_max = 1.0
    val_min = 0.0
    val_max = 0.1

    h = np.logical_and(
        (hsv_img[:, :, 0] >= hue_min),
        (hsv_img[:, :, 0] <= hue_max))

    BW = h.astype("uint8")

    return BW


def get_specific_channel(image, channel_type, blur_kernel=None, blur_type="gaussian"):

        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        if blur_kernel:
            if blur_type == "gaussian":
                image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
            elif blur_type == "median":
                image = cv2.medianBlur(image, blur_kernel)
            elif blur_type == "bilateral":
                image = cv2.bilateralFilter(image, blur_kernel, 75, 75)

        if channel_type == "blue":
            blue, green, red = cv2.split(image)
            return blue
        elif channel_type == "green":
            blue, green, red = cv2.split(image)
            return green
        elif channel_type == "red":
            blue, green, red = cv2.split(image)
            return red

        elif channel_type == "hue":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue, sat, val = cv2.split(image)
            return hue
        elif channel_type == "sat":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue, sat, val = cv2.split(image)
            return sat
        elif channel_type == "val":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue, sat, val = cv2.split(image)
            return val

        elif channel_type == "l_star*":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_star, a_star, b_star = cv2.split(image)
            return l_star
        elif channel_type == "a_star*":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_star, a_star, b_star = cv2.split(image)
            return a_star
        elif channel_type == "b_star":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_star, a_star, b_star = cv2.split(image)
            return b_star

        elif channel_type == "b_star_hist_equal":
            b, g, r = cv2.split(image)
            equ_b = cv2.equalizeHist(b)
            equ_g = cv2.equalizeHist(g)
            equ_r = cv2.equalizeHist(r)
            image = cv2.merge([equ_b, equ_g, equ_r])

            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_star, a_star, b_star = cv2.split(image)
            return b_star


        elif channel_type == "gray":
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        elif channel_type == "gray_hist_equal":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_star, a_star, b_star = cv2.split(image)
            return b_star

        elif channel_type == "chroma":
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            chroma = np.sqrt((a*a) + (b*b))
            return chroma
        elif channel_type == "maxrgb_chroma":
            image = max_rgb_filter(image)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            chroma = np.sqrt((a*a) + (b*b))
            return chroma

        elif channel_type == "smrgb_gray":
            image = special_max_r_gb_filter(image)
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif channel_type == "smrgb_hue":
            image = special_max_r_gb_filter(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue, sat, val = cv2.split(image)
            return hue

        elif channel_type == "maxrgb_gray":
            image = max_rgb_filter(image)
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif channel_type == "maxrgb_hue":
            image = max_rgb_filter(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue, sat, val = cv2.split(image)
            return hue
        elif channel_type == "maxrgb_b_star":
            image = max_rgb_filter(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_star, a_star, b_star = cv2.split(image)
            return b_star

        else:
            raise "Unsupported channel type {}".format(channel_type)


