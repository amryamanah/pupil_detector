import os
import cv2
from skimage.filters import threshold_adaptive
import csv
from livestockwatch.pupil_finder import PupilFinder
from livestockwatch.descriptors import HogDescriptor
from livestockwatch.utils import Rectangle, special_max_r_gb_filter, max_rgb_filter, adjust_gamma, get_blue_ratio
from IPython import embed
import numpy as np
import imutils
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


orientation = 9
ppc = 4
cpb = 1
result_path = os.path.join("result", "bbox_img")
kernel_type = "linear"
blur_kernel = 15
blur_type = "bilateral"
color_channel = "b_star"

svm_classifier_path = os.path.join("hog_model",
                                   "{}".format(kernel_type),
                                   "o{}_ppc{}_cpb{}".format(orientation, ppc, cpb),
                                   "{}".format(color_channel),
                                   "hog_svm.pkl")
# win_size = 256
# step_size = 64
# img_width = 1280
# img_height = 960
#
#
# descriptor = HogDescriptor(orientation=orientation, pixels_per_cell=ppc, cells_per_block=cpb)
# pupil_finder = PupilFinder(descriptor, svm_classifier_path=svm_classifier_path,
#                            win_size=win_size, step_size=step_size,
#                            img_width=img_width, img_height=img_height,
#                            channel_type=color_channel, blur_kernel=blur_kernel, blur_type=blur_type,
#                            svm_kernel_type=kernel_type)
#
data_folder = "test_pupil"


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

for dirpath, dirnames, files in os.walk(data_folder):
    for filename in files:
        if len(dirpath.split(os.path.sep)) == 1:
            if (filename.startswith("pl") or filename.startswith("nopl")) and filename.endswith(".bmp"):
                if filename.startswith("pl"):
                    acquisition_type = "pl"
                elif filename.startswith("nopl"):
                    acquisition_type = "nopl"

                result_folder = os.path.join(dirpath, "result", filename.split(".")[0])
                os.makedirs(result_folder, exist_ok=True)

                img_path = os.path.join(dirpath, filename)
                pupil_bbox = cv2.imread(img_path)
                pupil_bbox = cv2.fastNlMeansDenoisingColored(pupil_bbox, None, 10, 10, 7, 21)
                pupil_bbox_untouched = pupil_bbox.copy()

                hsv = cv2.cvtColor(pupil_bbox, cv2.COLOR_BGR2HSV)
                (hue, sat, val) = cv2.split(hsv)
                (_, mask) = cv2.threshold(sat, 30, 255, cv2.THRESH_BINARY)
                pupil_bbox = cv2.bitwise_and(pupil_bbox, pupil_bbox, mask=mask)

                # for blur_kernel in range(23, 50, 2):
                #     blurred = cv2.bilateralFilter(pupil_bbox, blur_kernel, 75, 75)
                #     cv2.putText(blurred, "BK = {}".format(blur_kernel),
                #                 (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)
                #     cv2.imshow("blur".format(blur_kernel), blurred)
                #     cv2.waitKey(0)
                blurred = cv2.bilateralFilter(pupil_bbox, 50, 75, 75)
                blurred = cv2.fastNlMeansDenoisingColored(blurred, None, 10, 10, 7, 21)
                blurred = cv2.medianBlur(blurred, 23)
                cv2.imwrite(os.path.join(result_folder, "blur.bmp"), blurred)
                # blurred = cv2.medianBlur(pupil_bbox, 23)
                hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
                (hue, sat, val) = cv2.split(hsv)
                image = blurred.reshape((blurred.shape[0] * blurred.shape[1], 3))

                clt = KMeans(n_clusters=4)
                clt.fit(image)
                result = clt.labels_.reshape((blurred.shape[0], blurred.shape[1]))
                # build a histogram of clusters and then create a figure
                # representing the number of pixels labeled to each color
                hist = centroid_histogram(clt)
                bar = plot_colors(hist, clt.cluster_centers_)
                # show our color bart
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)
                ax1.imshow(blurred, interpolation='nearest')
                ax1.axis('off')
                ax1.set_adjustable('box-forced')
                ax2.imshow(result, interpolation='nearest')
                ax2.axis('off')
                ax2.set_adjustable('box-forced')

                plt.show()


            #

            #
            #     lab_pupil = cv2.cvtColor(pupil_bbox, cv2.COLOR_BGR2LAB)
            #     l_star, a_star, b_star = cv2.split(lab_pupil)
            #
            #     hsv = cv2.cvtColor(pupil_bbox, cv2.COLOR_BGR2HSV)
            #     (hue, sat, val) = cv2.split(hsv)
            #     gray = cv2.cvtColor(pupil_bbox, cv2.COLOR_BGR2GRAY)
            #
            #     blue_ratio = get_blue_ratio(pupil_bbox)
            # #
            #     # equ = cv2.equalizeHist(gray)
            #
            #     blur_kernel = 11
            #     for blur_kernel in range(3, 24, 2):
            #         blurred = cv2.bilateralFilter(gray, blur_kernel, 75, 75)
            #
            #         sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
            #         sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
            #         edge = np.sqrt((np.power(sobelx, 2) + np.power(sobely, 2)))
            #
            #         final_folder = os.path.join(result_folder, "blur{}".format(blur_kernel))
            #         os.makedirs(final_folder, exist_ok=True)
            #         cv2.imwrite(os.path.join(final_folder, "blur.bmp"), blurred)
            #         cv2.imwrite(os.path.join(final_folder, "edge.bmp"), edge)
            #         cv2.imwrite(os.path.join(final_folder, "blue_ratio.bmp"), blue_ratio)
            #
            #         cv2.imwrite(os.path.join(final_folder, "original.bmp"), pupil_bbox_untouched)
            #         cv2.imwrite(os.path.join(final_folder, "gamma.bmp"), adjust_gamma(pupil_bbox_untouched, 2.0))
            #
            #         cv2.imwrite(os.path.join(final_folder, "preprocess.bmp"), pupil_bbox)
            #         for i in np.arange(0, 3, 0.1):
            #             thresh = imutils.auto_canny(blurred.copy(), sigma=i)
            #             cv2.imwrite(os.path.join(final_folder, "thresh{}.bmp".format(i)), thresh)







