import os
import cv2
import shutil

import numpy as np
from livestockwatch.pupil_finder import PupilFinder
from livestockwatch.descriptors import HogDescriptor
from livestockwatch.utils import is_skipped_folder, equ_hist_color_image, write_as_png, is_too_dark

from IPython import embed

orientation = 9
ppc = 4
cpb = 1
kernel_type = "linear"
blur_kernel = 15
blur_type = "bilateral"
# color_channel = "b_star"
color_channel = "b_star_hist_equal"

# for color_channel in ["gray", "b_star"]:

svm_classifier_path = os.path.join("hog_model",
                                   "{}".format(kernel_type),
                                   "o{}_ppc{}_cpb{}".format(orientation, ppc, cpb),
                                   "{}".format(color_channel),
                                   "hog_svm.pkl")
win_size = 256
step_size = 64
img_width = 1280
img_height = 960

video_show = False

descriptor = HogDescriptor(orientation=orientation, pixels_per_cell=ppc, cells_per_block=cpb)
pupil_finder = PupilFinder(descriptor, svm_classifier_path=svm_classifier_path,
                           win_size=win_size, main_step_size=step_size,
                           img_width=img_width, img_height=img_height,
                           channel_type=color_channel, blur_kernel=blur_kernel, blur_type=blur_type,
                           svm_kernel_type=kernel_type,
                           debug=True, extent_pixel=150)

# analysis_type = "full"
analysis_type = "partial"
max_timestamp = 2.0
step_second = 1.0

skip_folder = ["pending", "finished",
               "nopl", "pl",
               "bbox_img", "candidate", "final",
               "hard_neg", "raw", "hist_equ", "hist_equ_blur", "positive", "raw", "union"]

scanned_folder = "data_analysis/hansan_data"
scanned_folder = os.path.abspath(scanned_folder)

for dirpath, dirnames, files in os.walk(scanned_folder):
    for filename in files:
        if filename.endswith(".avi"):
            acquisition_type = "nopl"
            nopl_result_path = os.path.join(dirpath, acquisition_type)
            # if os.path.exists(nopl_result_path):
            #     shutil.rmtree(nopl_result_path)
            os.makedirs(nopl_result_path, exist_ok=True)
            os.makedirs(os.path.join(nopl_result_path, "raw"), exist_ok=True)
            video_path = os.path.join(dirpath, filename)
            print("Start processing {}".format(video_path))
            video = cv2.VideoCapture(video_path)
            first_eye_timestamp = None
            lst_frame = None
            frame_count = 0
            while True:
                (grabbed, frame) = video.read()

                if not grabbed:
                    break

                frame_timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.00
                frame_timestamp = float("{:.2f}".format(frame_timestamp))
                frame_count += 1
                # frame = cv2.resize(frame, (1280, 960))

                if frame_timestamp < 2.1:
                    write_as_png(os.path.join(nopl_result_path,"raw", "{}.png".format(frame_timestamp)), frame)
                    # final_eye_flag, final_point, final_patch_equ = pupil_finder.detect_pupil(
                    #         frame, nopl_result_path,
                    #         frame_timestamp)
                    # if final_eye_flag:
                    #     print("Set First frame timestamp at {}s".format(frame_timestamp))
                    #     first_eye_timestamp = frame_timestamp
                    #     lst_frame = np.arange(first_eye_timestamp, max_timestamp, step_second)

            video.release()
            print("Finish processing {}".format(video_path))
            print(frame_count)
