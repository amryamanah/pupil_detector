import os
import cv2
from livestockwatch.utils import write_as_png

scanned_folder = os.path.join("data_analysis", "TA02")

for dirpath, dirnames, files in os.walk(scanned_folder):
    for filename in files:
        if filename.endswith(".bmp"):
            img = cv2.imread(os.path.join(dirpath, filename))
            if filename.startswith("mask"):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                print(img.shape)
            filename_part = os.path.splitext(filename)
            new_filename = "{}.png".format(filename_part[0])
            write_as_png(os.path.join(dirpath, new_filename), img)

for dirpath, dirnames, files in os.walk(scanned_folder):
    for filename in files:
        if filename.endswith(".bmp"):
            os.remove(os.path.join(dirpath, filename))