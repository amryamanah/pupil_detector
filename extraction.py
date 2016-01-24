import cv2
import argparse
import os
from IPython import embed
from livestockwatch.fitellipse import *

from livestockwatch.pupil_analysis import pupil_fit_ellipse


SKIP_FOLDER = ["pending", "finished",
               "nopl", "pl",
               "bbox_img", "candidate", "final", "plr_result", "final_extended",
               "hard_neg", "raw", "hist_equ", "hist_equ_blur", "positive", "raw", "union"]


def contour_iterator(contour):
    while contour:
        yield contour
        contour = contour.h_next()


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help= "Path to the scanned folder")
    args = vars(ap.parse_args())
    print(args)

    scanned_folder = os.path.abspath(args["input"])

    for dirpath, dirnames, files in os.walk(scanned_folder):
        part_dirpath = dirpath.split(os.sep)
        if part_dirpath[-1] == "final_extended":
            print(dirpath)
            for filename in files:
                if filename.endswith(".png"):
                    print(filename)
                    img_path = os.path.join(dirpath, filename)
                    image = cv2.imread(img_path)
                    major_axis, minor_axis, image_untouched = pupil_fit_ellipse(image)


        # if is_skipped_folder(dirpath, SKIP_FOLDER):
        #     print("[SKIPPED] dirpath {}".format(dirpath))
        # else:
        #     print("[PROCESSED] dirpath {}".format(dirpath))
        #     for filename in files:
        #         if filename.endswith(".png"):
        #             print(filename)
        #             img_path = os.path.join(dirpath, filename)
        #             image = cv2.imread(img_path)
        #             get_pupil(image)




