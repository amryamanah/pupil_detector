import os
import cv2
import shutil
import argparse
import asyncio

import numpy as np
from livestockwatch.pupil_finder import PupilFinder
from livestockwatch.descriptors import HogDescriptor
from livestockwatch.utils import is_skipped_folder

from IPython import embed


orientation = 9
ppc = 4
cpb = 1
kernel_type = "linear"
blur_kernel = 15
blur_type = "bilateral"
color_channel = "b_star_hist_equal"
# color_channel = "b_star"
svm_classifier_path = os.path.join("hog_model",
                                   "{}".format(kernel_type),
                                   "o{}_ppc{}_cpb{}".format(orientation, ppc, cpb),
                                   "{}".format(color_channel),
                                   "hog_svm.pkl")
debug = False
win_size = 256
step_size = 80
img_width = 1280
img_height = 960
analysis_type = "partial"
max_timestamp = 2.0
step_second = 1.0

loop = asyncio.get_event_loop()

skip_folder = ["pending", "finished",
               "nopl", "pl",
               "bbox_img", "candidate", "final", "plr_result", "final_extended",
               "hard_neg", "raw", "hist_equ", "hist_equ_blur", "positive", "raw", "union"]
descriptor = HogDescriptor(orientation=orientation, pixels_per_cell=ppc, cells_per_block=cpb)
pupil_finder = PupilFinder(loop=loop,
                           descriptor=descriptor, svm_classifier_path=svm_classifier_path,
                           win_size=win_size, step_size=step_size,
                           img_width=img_width, img_height=img_height,
                           channel_type=color_channel, blur_kernel=blur_kernel, blur_type=blur_type,
                           svm_kernel_type=kernel_type,
                           debug=debug)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help= "Path to the scanned folder")
args = vars(ap.parse_args())

scanned_folder = os.path.abspath(args["input"])

for dirpath, dirnames, files in os.walk(scanned_folder):
    part_dirpath = dirpath.split(os.sep)
    if is_skipped_folder(dirpath, skip_folder):
        print("[SKIPPED] dirpath {}".format(dirpath))
    else:
        print("[PROCESSED] dirpath {}".format(dirpath))
        for filename in files:
            if filename.endswith(".avi"):
                acquisition_type = "nopl"
                nopl_result_path = os.path.join(dirpath, acquisition_type)
                if os.path.exists(nopl_result_path):
                    shutil.rmtree(nopl_result_path)
                os.makedirs(nopl_result_path, exist_ok=True)
                video_path = os.path.join(dirpath, filename)
                print("Start processing {}".format(video_path))
                video = cv2.VideoCapture(video_path)
                first_eye_timestamp = None
                lst_frame = None
                frame_count = 1
                while True:
                    (grabbed, frame) = video.read()

                    if not grabbed:
                        break

                    frame_timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.00
                    if frame_timestamp == 0.0:
                        frame_timestamp = frame_count * 0.06666666666666667

                    frame_timestamp = float("{:.2f}".format(frame_timestamp))
                    filename = "{}_{}".format(frame_count, frame_timestamp)

                    print(filename)
                    if not first_eye_timestamp and frame_timestamp > 0.5:
                        final_eye_flag, final_point, final_patch_equ = pupil_finder.detect_pupil(
                                frame, nopl_result_path,
                                filename)
                        if final_eye_flag:
                            print("Set First frame timestamp at {}s".format(frame_timestamp))
                            first_eye_timestamp = frame_timestamp
                            lst_frame = np.arange(first_eye_timestamp, max_timestamp, step_second)

                    elif first_eye_timestamp:
                        if frame_timestamp < 3.0:
                            print("Analyzing frame_timestamp {}".format(frame_timestamp))
                            final_eye_flag, final_point, final_patch_equ = \
                                    pupil_finder.detect_pupil(
                                            frame, nopl_result_path,
                                            filename
                                )
                    frame_count += 1

                video.release()
                print("Finish processing {}".format(video_path))


                    # elif (filename.startswith("pl") or filename.startswith("no") or filename.startswith("nopl")) \
                    #         and filename.endswith("bmp"):
                    #     if filename.startswith("pl"):
                    #         acquisition_type = "pl"
                    #     elif filename.startswith("nopl") or filename.startswith("no"):
                    #         acquisition_type = "nopl"
                    #
                    #     pl_result_path = os.path.join(dirpath, "{}".format(kernel_type),
                    #                                   "o{}_ppc{}_cpb{}".format(orientation, ppc, cpb),
                    #                                   color_channel, acquisition_type)
                    #     os.removedirs(pl_result_path)
                    #     os.makedirs(pl_result_path, exist_ok=True)
                    #     img_path = os.path.join(dirpath, filename)
                    #     img = cv2.imread(img_path)
                    #     final_point, result_frame = pupil_finder.detect_pupil(img, pl_result_path, filename)

loop.close()