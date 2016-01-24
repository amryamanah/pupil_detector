import os
import argparse
import asyncio
from pprint import pprint
from time import time

from livestockwatch.pupil_finder import main_detector
from livestockwatch.utils import is_skipped_folder
from livestockwatch.config import SKIP_FOLDER
from IPython import embed

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Path to the scanned folder")
ap.add_argument("-m", "--main-step", required=False, type=int, default=64, help="Main step")
ap.add_argument("-s", "--secondary-step", required=False, type=int, default=8, help="Secondary step")
ap.add_argument("-d", "--duration", required=False, type=float, default=3.0, help="duration of video to be analyzed")
ap.add_argument("--debug", required=False, type=bool, default=False, help="debug flag")
args = vars(ap.parse_args())
print(args)


# mongo_url = "mongodb://biseadmin:bise2014@162.243.1.142:27017/livestockwatch"
# mongo_url = "mongodb://127.0.0.1:27017/livestockwatch"
# orientation = 9
# ppc = 4
# cpb = 1
# kernel_type = "linear"
# blur_kernel = 15
# blur_type = "bilateral"
# color_channel = "b_star_hist_equal"
# svm_classifier_path = os.path.join("hog_model",
#                                    "{}".format(kernel_type),
#                                    "o{}_ppc{}_cpb{}".format(orientation, ppc, cpb),
#                                    "{}".format(color_channel),
#                                    "hog_svm.pkl")
# debug = args["debug"]
# win_size = 256
# img_width = 1280
# img_height = 960
# analysis_type = "partial"
# max_timestamp = 2.0
# step_second = 1.0
#
#
#
# skip_folder = ["pending", "finished",
#                "nopl", "pl",
#                "bbox_img", "candidate", "final", "plr_result", "final_extended",
#                "hard_neg", "raw", "hist_equ", "hist_equ_blur", "positive", "raw", "union"]

# descriptor = HogDescriptor(orientation=orientation, pixels_per_cell=ppc, cells_per_block=cpb)
# pupil_finder = PupilFinder(
#                            descriptor=descriptor, svm_classifier_path=svm_classifier_path, win_size=win_size,
#                            main_step_size=args["main_step"], secondary_step_size=args["secondary_step"],
#                            img_width=img_width, img_height=img_height,
#                            channel_type=color_channel, blur_kernel=blur_kernel, blur_type=blur_type,
#                            svm_kernel_type=kernel_type,
#                            debug=debug)

scanned_folder = os.path.abspath(args["input"])
lst_job = []
for dirpath, dirnames, files in os.walk(scanned_folder):
    part_dirpath = dirpath.split(os.sep)
    if is_skipped_folder(dirpath, SKIP_FOLDER):
        # print("[SKIPPED] dirpath {}".format(dirpath))
        pass
    else:
        # print("[PROCESSED] dirpath {}".format(dirpath))
        for filename in files:
            if filename.endswith(".avi"):
                job_kwargs = {
                    "dirpath": dirpath,
                    "filename": filename,
                    "debug": args["debug"],
                    "duration": args["duration"],
                    "main_step": args["main_step"],
                    "secondary_step": args["secondary_step"]
                }
                lst_job.append(job_kwargs)

start_time = time()
pprint("[START] Processing {} capture session".format(len(lst_job)))
# Parallel(n_jobs=4)(delayed(main_detector)(**job) for job in lst_job)
for job in lst_job:
    main_detector(**job)

pprint("[END] Processing {} capture session done in {:.3f} minute".format(
        len(lst_job), (time() - start_time) / 60))
loop = asyncio.get_event_loop()
loop.close()
