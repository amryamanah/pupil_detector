import os
import csv
import cv2
import uuid
from sklearn.externals import joblib
from livestockwatch.simple_classifier import SimpleClassifier
from livestockwatch.descriptors import HogDescriptor
from IPython import embed

blur_kernel = 15
blur_type = "bilateral"
orientation = 9
ppc = 4
cpb = 1
lst_color_channel = ["b_star", "gray"]
color_channel = "b_star_hist_equal"


descriptor = HogDescriptor(orientation=orientation, pixels_per_cell=ppc, cells_per_block=cpb)
classifier = SimpleClassifier("dataset", "hog_model",
                              color_channel, blur_kernel, blur_type, descriptor,
                              kernel_type="linear", svm_implementation="default")
classifier.load_dataset()
classifier.training()

# for color_channel in lst_color_channel:
#     descriptor = HogDescriptor(orientation=orientation, pixels_per_cell=ppc, cells_per_block=cpb)
#     classifier = SimpleClassifier("dataset_oct", "hog_model", negdata_size, posdata_size,
#                                   color_channel, blur_kernel, blur_type, descriptor,
#                                   kernel_type="linear", svm_implementation="default")
#     classifier.load_dataset()
#     classifier.training()

# # Testing best orientation and cpb and ppc.
# for cpb in [1, 2, 3, 4]:
#     for ppc in range(4, 13):
#         for orientation in [9, 8, 7, 6, 5]:
#             for color_channel in ["b_star", "maxrgb_b_star", "gray"]:
#                 descriptor = HogDescriptor(orientation=orientation, pixels_per_cell=ppc, cells_per_block=cpb)
#                 classifier = SimpleClassifier("dataset_oct", "hog_model", negdata_size, posdata_size,
#                                               color_channel, blur_kernel, blur_type, descriptor,
#                                               kernel_type="linear", svm_implementation="default")
#                 classifier.load_dataset()
#                 classifier.training()

# Testing best combination blur_method and blur_kernel
# for blur_kernel in range(3, 76, 6):
#     for blur_type in ["gaussian", "median", "bilateral"]:
#         descriptor = HogDescriptor(orientations=9, pixels_per_cell=(12, 12), cells_per_block=(3, 3))
#         classifier = SimpleClassifier("dataset_oct", "testing_blur", 260, 118,
#                                       "gray", blur_kernel, blur_type, descriptor,
#                                       kernel_type="linear", svm_implementation="libsvm")
#         classifier.load_dataset()
#         classifier.training()


# # Testing best kernel type
# blur_kernel = 15
# blur_type = "bilateral"
# ppc = 12
# cpb = 3
#
# for kernel_type in ["sigmoid", "linear", "rbf", "poly"]:
#     descriptor = HogDescriptor(orientations=9, pixels_per_cell=(ppc, ppc), cells_per_block=(cpb, cpb))
#     classifier = SimpleClassifier("dataset_oct", "testing_kernel_type", 260, 118,
#                                   "gray", blur_kernel, blur_type, descriptor,
#                                   kernel_type=kernel_type, svm_implementation="default")
#     classifier.load_dataset()
#     classifier.training()


# # Testing best svm implementation
# blur_kernel = 15
# blur_type = "bilateral"
# ppc = 12
# cpb = 3
#
# for svm_implementation in ["liblinear", "libsvm", "default"]:
#     descriptor = HogDescriptor(orientations=9, pixels_per_cell=(ppc, ppc), cells_per_block=(cpb, cpb))
#     classifier = SimpleClassifier("dataset_oct", "testing_svm_implementation", 248, 118,
#                                   "gray", blur_kernel, blur_type, descriptor,
#                                   kernel_type="linear", svm_implementation=svm_implementation)
#     classifier.load_dataset()
#     classifier.training()

# # Testing best svm implementation
# blur_kernel = 15
# blur_type = "bilateral"
# ppc = 12
# cpb = 3
#
# for color_space in ["hue", "maxrgb_chroma", "maxrgb_b_star", "maxrgb_gray", "maxrgb_hue", "b_star", "blue", "chroma", "gray"]:
#     descriptor = HogDescriptor(orientations=9, pixels_per_cell=(ppc, ppc), cells_per_block=(cpb, cpb))
#     classifier = SimpleClassifier("dataset_oct", "testing_color_channel", 248, 118,
#                                   color_space, blur_kernel, blur_type, descriptor,
#                                   kernel_type="linear", svm_implementation="default")
#     classifier.load_dataset()
#     classifier.training()

